"""Render saved contamination-study figures; never generate data or fit models."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / 'docs/research/rare_subgroups/contamination_v1'
STYLES = {
    'NORES': ('No resampling','#666666','o','--'),
    'WEIGHTED': ('Class weighting','#111111','s',':'),
    'ROS': ('ROS','#0072B2','^','-'),
    'SMOTE': ('SMOTE','#009E73','D','-'),
    'AW_REMOVE': ('AW removal','#D55E00','X','-'),
    'AW_SEED_FILTER_RETAIN': ('AW retention control','#E69F00','v','-'),
    'AW_REMOVE_TUNED': ('AW tuned','#CC79A7','P','-'),
}


def plot_scores(scores: Path, output: Path) -> list[Path]:
    frame = pd.read_csv(scores)
    expected = {(r,s,m,seed,method) for r in (4,8,16) for s in (.35,.90)
                for m in (0,5,10) for seed in range(42,47) for method in STYLES}
    actual = set(frame[['fit_rare_count','rare_sigma','flipped_count','seed','method']].itertuples(index=False,name=None))
    if len(frame)!=630 or actual!=expected or not frame.status.eq('success').all():
        raise ValueError('These figures require all 630 successful rows of the frozen study.')
    numeric = ['rare_recall','common_recall','test_fpr','f1','ap','rare_retained_fraction','common_retained_fraction']
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError('Successful plot metrics must be finite.')
    if not np.isfinite(frame.loc[frame.flipped_count>0,'noise_retained_fraction']).all():
        raise ValueError('Noise retention must be defined for contaminated cells.')
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figures=[]; paths=[]
    fig,axes=plt.subplots(2,3,figsize=(14.5,8.8),sharex=True,sharey=True)
    handles=[]
    for row,sigma in enumerate((.35,.90)):
        for col,r in enumerate((4,8,16)):
            ax=axes[row,col]
            data=frame[(frame.fit_rare_count==r)&(frame.rare_sigma==sigma)]
            for method,(label,color,marker,style) in STYLES.items():
                subset=data[data.method==method]
                means=subset.groupby('flipped_count').rare_recall.mean().sort_index()
                for m,group in subset.groupby('flipped_count'):
                    ax.scatter(m+np.linspace(-.12,.12,5),group.sort_values('seed').rare_recall,
                               color=color,alpha=.18,s=12,edgecolors='none',zorder=2)
                line,=ax.plot(means.index,means,color=color,marker=marker,linestyle=style,
                              linewidth=1.7,markersize=5,label=label,zorder=3)
                if row==0 and col==0: handles.append(line)
            ax.set_title(f'Rare fitting count = {r}; spread = {sigma:.2f}',fontsize=11,weight='bold')
            ax.set_xticks([0,5,10]); ax.set_ylim(0,1.04); ax.grid(axis='y',alpha=.2)
            if row==1: ax.set_xlabel('Wrong training labels (out of 1,100)')
            if col==0: ax.set_ylabel('Rare-group recall')
    fig.suptitle('Rare-group recall under training-label contamination',fontsize=18,weight='bold',y=.98)
    fig.text(.5,.93,'Same feature rows and true groups across noise levels. Lines: five-seed means; faint points: seeds.',ha='center')
    fig.legend(handles=handles,labels=[h.get_label() for h in handles],ncol=4,loc='lower center',bbox_to_anchor=(.5,.035),frameon=False)
    fig.text(.5,.012,'RF, 100 trees; calibration FPR = 0.05 for every model. Actual test FPR varies and is reported separately.',ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(top=.86,bottom=.20,left=.065,right=.99,hspace=.27,wspace=.13)
    path=output/'rare_recall_by_noise.png';fig.savefig(path,dpi=170);paths.append(path);figures.append(fig)

    fig,axes=plt.subplots(2,3,figsize=(14.5,8.8),sharex=True,sharey=True)
    default=frame[frame.method=='AW_REMOVE']
    categories=[('common_retained_fraction','Valid common','#777777'),('rare_retained_fraction','Valid rare','#D55E00'),('noise_retained_fraction','Mislabeled majority','#756BB1')]
    for row,sigma in enumerate((.35,.90)):
        for col,r in enumerate((4,8,16)):
            ax=axes[row,col]
            data=default[(default.fit_rare_count==r)&(default.rare_sigma==sigma)]
            for k,(metric,label,color) in enumerate(categories):
                values=data[data.flipped_count>0].groupby('flipped_count')[metric].mean().sort_index()
                offset=(k-1)*.25
                ax.bar(np.arange(2)+offset,values,width=.23,color=color,label=label,alpha=.85)
                for i,m in enumerate((5,10)):
                    seed_values=data[data.flipped_count==m].sort_values('seed')[metric]
                    ax.scatter(i+offset+np.linspace(-.03,.03,5),seed_values,s=13,color='black',alpha=.35,zorder=3)
            ax.set_title(f'Rare fitting count = {r}; spread = {sigma:.2f}',fontsize=11,weight='bold')
            ax.set_xticks([0,1],['5','10']); ax.set_ylim(0,1.08); ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            if row==1: ax.set_xlabel('Wrong training labels (out of 1,100)')
            if col==0: ax.set_ylabel('Fraction of original group retained')
    fig.suptitle('Which originals did default AW removal retain?',fontsize=18,weight='bold',y=.98)
    fig.text(.5,.93,'Bars: five-seed means; dots: individual seeds. Ground-truth identities are evaluator-only.',ha='center')
    fig.legend(*axes[0,0].get_legend_handles_labels(),ncol=3,loc='lower center',bbox_to_anchor=(.5,.04),frameon=False)
    fig.text(.5,.012,'Retained mislabeled originals are one diagnostic; synthetic parent exposure and final training counts are saved separately.',ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(top=.82,bottom=.18,left=.065,right=.99,hspace=.34,wspace=.13)
    path=output/'original_retention_by_noise.png';fig.savefig(path,dpi=170);paths.append(path);figures.append(fig)

    fig,ax=plt.subplots(figsize=(13.6,5.7))
    indexed=frame.set_index(['fit_rare_count','rare_sigma','flipped_count','seed','method'])
    base=[(r,sigma) for r in (4,8,16) for sigma in (.35,.90)]
    all_values=[]
    for m,offset,color,marker in [(5,-.12,'#0072B2','o'),(10,.12,'#CC79A7','s')]:
        means=[]
        for i,(r,sigma) in enumerate(base):
            values=[]
            for seed in range(42,47):
                at=lambda noise,method:float(indexed.loc[(r,sigma,noise,seed,method),'ap'])
                values.append((at(m,'AW_SEED_FILTER_RETAIN')-at(0,'AW_SEED_FILTER_RETAIN'))-(at(m,'AW_REMOVE')-at(0,'AW_REMOVE')))
            means.append(np.mean(values));all_values.extend(values)
            ax.scatter(i+offset+np.linspace(-.035,.035,5),values,color=color,alpha=.35,s=25)
        ax.scatter(np.arange(6)+offset,means,color=color,marker=marker,s=95,edgecolors='black',linewidths=.5,label=f'{m} wrong labels: mean',zorder=4)
    ax.axhline(0,color='#444444',linewidth=1)
    ax.axhline(-.02,color='#777777',linestyle='--',linewidth=1,label='Predeclared mean cutoff: -0.02')
    ax.set_xticks(range(6),[f'r={r}, spread={s:.2f}' for r,s in base])
    ax.set_ylabel('Paired AP interaction: retention minus removal')
    ax.grid(axis='y',alpha=.2)
    fig.suptitle('Did retention lose more AP under contamination?',fontsize=18,weight='bold',y=.98)
    fig.text(.5,.91,'Interaction = (retention noisy - clean AP) - (removal noisy - clean AP)',ha='center')
    ax.legend(loc='best',frameon=False)
    fig.text(.5,.04,'Below zero: retention lost more AP. Markers: five-seed means; faint dots: seeds. The full screen also requires rare-recall and FPR conditions.',ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(top=.80,bottom=.17,left=.085,right=.99)
    path=output/'contamination_AP_interaction.png';fig.savefig(path,dpi=170);paths.append(path);figures.append(fig)
    path=output/'Rare_Subgroup_Contamination_Figures.pdf'
    with PdfPages(path,metadata={'CreationDate':None,'ModDate':None}) as pdf:
        for figure in figures:pdf.savefig(figure)
    for figure in figures:plt.close(figure)
    return [*paths,path]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores',type=Path,default=PUBLISHED/'scores.csv')
    parser.add_argument('--output',type=Path,default=PUBLISHED/'figures')
    args=parser.parse_args()
    for path in plot_scores(args.scores,args.output):print(path)


if __name__=='__main__':main()
