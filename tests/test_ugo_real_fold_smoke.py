"""Provider validation failure behavior on artificial files, not real evidence."""
import hashlib

import pytest

from scripts.ugo_real_fold_smoke import verify


def source_fixture(tmp_path):
    files = []
    rows = [(i, 'positive' if i % 4 == 0 else 'negative') for i in range(20)]
    for name in ['ecoli2', 'yeast3', 'ecoli4']:
        directory = tmp_path / 'data/raw/keel' / name
        directory.mkdir(parents=True)
        partitions = {name + '.dat': rows}
        for fold in range(1, 6):
            test = rows[4 * (fold - 1):4 * fold]
            partitions[f'{name}-5-{fold}tra.dat'] = [r for r in rows if r not in test]
            partitions[f'{name}-5-{fold}tst.dat'] = test
        for filename, observations in partitions.items():
            path = directory / filename
            text = ('@relation toy\n@attribute x real\n@attribute Class {negative,positive}\n'
                    '@inputs x\n@outputs Class\n@data\n')
            path.write_text(text + ''.join(f'{x},{y}\n' for x, y in observations))
            files.append(dict(path=str(path.relative_to(tmp_path)), url='https://unused.invalid',
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(files=files)


def test_fivefold_coverage_and_checksum_failure(tmp_path):
    manifest = source_fixture(tmp_path)
    result = verify(manifest, tmp_path)
    assert set(result) == {'ecoli2', 'yeast3', 'ecoli4'}
    assert all(r['folds'] == 5 and r['rows'] == 20 for r in result.values())
    path = tmp_path / manifest['files'][0]['path']
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='changed provider'):
        verify(manifest, tmp_path)


def test_matching_hash_does_not_replace_multiset_validation(tmp_path):
    manifest = source_fixture(tmp_path)
    item = next(r for r in manifest['files'] if r['path'].endswith('ecoli2-5-1tst.dat'))
    path = tmp_path / item['path']
    path.write_text(path.read_text().replace('1,negative', '99,negative'))
    item['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='multiset mismatch'):
        verify(manifest, tmp_path)


def test_missing_inputs_and_escaping_manifest_are_rejected(tmp_path):
    manifest = source_fixture(tmp_path)
    (tmp_path / manifest['files'][0]['path']).unlink()
    with pytest.raises(ValueError, match='Missing'):
        verify(manifest, tmp_path)
    with pytest.raises(ValueError, match='escapes'):
        verify(dict(files=[dict(path='../escape.dat')]), tmp_path)
