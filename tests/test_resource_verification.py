"""Offline import/fold checks; fixtures are artificial, never benchmark evidence."""
from pathlib import Path
import zipfile

import pytest
from scripts.import_resources import ImportError, import_dataset, safe_extract, sha256
from scripts.verify_keel import audit_folds, load_with_mapping, verify
from src.data_loader import DatasetError


def archive(path, names):
    with zipfile.ZipFile(path, 'w') as z:
        for name, value in names.items():z.writestr(name, value)
    return path


def dat(path, rows):
    path.write_text('@relation toy\n@attribute x real\n@attribute class {negative,positive}\n@inputs x\n@outputs class\n@data\n'+''.join(f'{x},{y}\n' for x,y in rows))
    return path


@pytest.mark.parametrize('name',['../escape.dat','/absolute.dat','C:/escape.dat','folder\\escape.dat','a/../../escape.dat'])
def test_reject_unsafe_paths_before_writing(tmp_path,name):
    z=archive(tmp_path/'bad.zip',{'safe.dat':'ok',name:'bad'})
    with pytest.raises(ImportError):safe_extract(z,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_symlink_archive_rejected(tmp_path):
    with zipfile.ZipFile(tmp_path/'bad.zip','w') as z:
        entry=zipfile.ZipInfo('link');entry.create_system=3;entry.external_attr=0o120777<<16;z.writestr(entry,'../target')
    with pytest.raises(ImportError,match='Symlink'):safe_extract(tmp_path/'bad.zip',tmp_path/'out')


def test_conflict_idempotency_and_source_hash(tmp_path):
    z=archive(tmp_path/'toy.zip',{'nested/original.dat':'original'})
    before=sha256(z);root=tmp_path/'project'
    first=import_dataset(z,'toy',root);second=import_dataset(z,'toy',root)
    assert first==second and sha256(z)==before
    first[0].write_text('user edit')
    with pytest.raises(ImportError,match='overwrite'):import_dataset(z,'toy',root)
    assert first[0].read_text()=='user edit'


@pytest.mark.parametrize('names',[{'a':'file','a/b':'child'},{'Data.dat':'one','data.dat':'two'}])
def test_internal_archive_conflicts(tmp_path,names):
    z=archive(tmp_path/'bad.zip',names)
    with pytest.raises(ImportError):safe_extract(z,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_limit_and_existing_parent_escape(tmp_path):
    z=archive(tmp_path/'data.zip',{'a/data':'12345'})
    with pytest.raises(ImportError,match='size'):safe_extract(z,tmp_path/'out',max_bytes=3)
    out=tmp_path/'out';out.mkdir();external=tmp_path/'external';external.mkdir()
    (out/'a').symlink_to(external,target_is_directory=True)
    with pytest.raises(ImportError,match='escape'):safe_extract(z,out)
    assert not list(external.iterdir())


def test_mapping_uses_training_roles_when_test_counts_reverse(tmp_path):
    test=dat(tmp_path/'test.dat',[(1,'negative'),(2,'positive'),(3,'positive')])
    d=load_with_mapping(test,{'negative':0,'positive':1})
    assert d.y.tolist()==[0,1,1] and d.minority_count==2


def test_unknown_labels_rejected(tmp_path):
    test=dat(tmp_path/'test.dat',[(1,'negative'),(2,'negative'),(3,'positive')])
    with pytest.raises(DatasetError,match='vocabularies'):
        load_with_mapping(test,{'negative':0,'other':1})


def make_folds(folder):
    folder.mkdir();observations=[(i,'positive' if i%4==0 else 'negative') for i in range(20)]
    full=dat(folder/'toy.dat',observations)
    for i in range(1,6):
        test=observations[(i-1)*4:i*4];train=[r for r in observations if r not in test]
        dat(folder/f'toy-5-{i}tra.dat',train);dat(folder/f'toy-5-{i}tst.dat',test)
    return full


def test_five_fold_multiset_coverage_and_incomplete_report(tmp_path):
    full=make_folds(tmp_path/'toy')
    report=audit_folds(full.parent,full)
    assert report['status']=='VERIFIED_MULTISET' and report['coverage'] is True
    assert all(r['identical_complete_occurrences']==0 for r in report['folds'])
    (full.parent/'toy-5-3tst.dat').unlink()
    report=audit_folds(full.parent,full)
    assert report['status']=='INVALID' and any('Incomplete' in e for e in report['errors'])


def test_fold_reuse_and_missing_folds(tmp_path):
    full=make_folds(tmp_path/'toy')
    (full.parent/'toy-5-2tst.dat').write_bytes((full.parent/'toy-5-1tst.dat').read_bytes())
    report=audit_folds(full.parent,full)
    assert report['status']=='INVALID' and report['coverage'] is False
    empty=tmp_path/'empty';empty.mkdir()
    assert audit_folds(empty,None)['status']=='MISSING_FOLDS'


def test_identical_values_are_reported_without_inventing_record_identity(tmp_path):
    folder=tmp_path/'toy';folder.mkdir()
    full=dat(folder/'toy.dat',[(0,'negative'),(0,'negative'),(1,'positive'),(2,'negative'),(3,'positive'),(4,'negative')])
    # Two-fold artificial fixture with a legitimate repeated value in both partitions.
    train=[(0,'negative'),(1,'positive'),(2,'negative')];test=[(0,'negative'),(3,'positive'),(4,'negative')]
    for i,tr,te in [(1,train,test),(2,test,train)]:
        dat(folder/f'toy-2-{i}tra.dat',tr);dat(folder/f'toy-2-{i}tst.dat',te)
    report=audit_folds(folder,full)
    assert report['status']=='VERIFIED_MULTISET'
    assert report['folds'][0]['identical_complete_occurrences']==1
    assert 'not distinguishable' in report['identity_limit']


@pytest.mark.parametrize('name',['ecoli2','glass6','yeast3','page-blocks0','ecoli4','yeast4','segment0','abalone19'])
def test_local_real_dataset_integration(name):
    from src.data_loader import load_dataset
    from src.imbalance_stats import calculate_statistics
    root=Path(__file__).resolve().parents[1]
    path=root/'data/raw/keel'/name/(name+'.dat')
    if not path.exists():pytest.skip('User-supplied raw resource is not available locally')
    # Independent count from original data lines, no feature preprocessing or fitting.
    text=path.read_text();data=text.lower().split('@data',1)[1]
    count=sum(bool(line.strip()) and not line.lstrip().startswith('%') for line in data.splitlines())
    d=load_dataset(path);s=calculate_statistics(d)
    assert s.total_samples==count and d.y.index.equals(d.X.index)
    assert set(d.label_mapping.values())=={0,1}
    assert s.majority_count+s.minority_count==count
    assert s.numerical_features+s.categorical_features==s.number_of_features
    assert d.original_y.map(d.label_mapping).equals(d.y)


def test_smoke_missing_folds_blocks_without_loading_full_dataset(tmp_path,monkeypatch):
    import scripts.ugo_smoke as smoke
    monkeypatch.setattr(smoke,'load_dataset',lambda *_:pytest.fail('Must not substitute full dataset'))
    out=tmp_path/'summary.csv'
    assert not smoke.run_smoke(tmp_path,{'ecoli2':{'status':'MISSING_FOLDS'}},['ecoli2'],out)
    assert out.read_text().count('BLOCKED_MISSING_VERIFIED_FIVE_FOLD')==2


def test_smoke_runner_never_opens_test_files(tmp_path,monkeypatch):
    import scripts.ugo_smoke as smoke
    full=make_folds(tmp_path/'toy');report=audit_folds(full.parent,full)
    report['folds']=report['folds'][:1]
    original=smoke.load_dataset;seen=[]
    def guarded(path):
        assert path.name.endswith('tra.dat');seen.append(path.name);return original(path)
    monkeypatch.setattr(smoke,'load_dataset',guarded)
    assert smoke.run_smoke(tmp_path,{'toy':report},['toy'],tmp_path/'summary.csv')
    assert seen==['toy-5-1tra.dat']
