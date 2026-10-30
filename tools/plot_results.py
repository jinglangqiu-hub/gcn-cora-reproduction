"""Generate traceable scientific figures from the accepted fixed-split GCN/MLP results."""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir.resolve()
    for sub in ['data','code','figures']:(out/sub).mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(out/'qa'/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    from matplotlib.ticker import MaxNLocator
    import hashlib
    def sha256(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    base=ROOT/'results'
    summary=json.loads((base/'test_summary.json').read_text())
    rows=read_csv(base/'test_runs.csv')
    assert len(rows)==20 and {(r['model'],int(r['seed'])) for r in rows}=={(m,s) for m in ['gcn','mlp'] for s in range(10)}
    for name in ['test_runs.csv','test_summary.json','paired_differences.csv','test_runs.csv']:
        shutil.copyfile(base/name,out/'data'/name)
    shutil.copyfile(Path(__file__),out/'code'/'plot_reproduction_results.py')
    font='DejaVu Sans'
    for candidate in [Path('C:/Windows/Fonts/msyh.ttc'),Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]:
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate));font=font_manager.FontProperties(fname=str(candidate)).get_name();break
    plt.rcParams.update({'font.family':font,'font.size':10,'axes.titlesize':11,
        'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,
        'svg.fonttype':'path','savefig.facecolor':'white','figure.facecolor':'white',
        'axes.edgecolor':'#66717e','axes.labelcolor':'#243447','text.color':'#243447'})
    blue,orange,gray='#245C91','#C57B22','#58616E'
    scores={(r['model'],int(r['seed'])):100*float(r['test_accuracy']) for r in rows}
    assets=[]
    def finish(fig,name,sources,note):
        for suffix in ['png','svg']:
            fig.savefig(out/'figures'/f'{name}.{suffix}',dpi=240,bbox_inches='tight')
        plt.close(fig)
        assets.append({'figure':name,'source_files':sources,'note':note,
            'outputs':{s:sha256(out/'figures'/f'{name}.{s}') for s in ['png','svg']}})
    def grid(ax):
        ax.grid(axis='y',color='#DCE2E8',linewidth=.65,alpha=.8);ax.set_axisbelow(True)
    fig,ax=plt.subplots(figsize=(7.6,4.1),layout='constrained')
    for x,(kind,color,ref) in enumerate([('gcn',blue,81.5),('mlp',orange,55.1)]):
        data=np.array([scores[(kind,s)] for s in range(10)])
        offsets=np.linspace(-.06,.09,10)
        ax.scatter(x+offsets,data,c=color,s=35,marker='o' if kind=='gcn' else 's',label='单次运行' if x==0 else None,zorder=4)
        mean,std=data.mean(),data.std(ddof=1)
        ax.errorbar(x-.24,mean,yerr=std,fmt='D',color=color,markersize=6,capsize=6,elinewidth=1.8,zorder=5)
        ax.scatter([x+.24],[ref],marker='^',facecolors='none',edgecolors=gray,s=85,zorder=5,label='论文固定划分均值（100 次）' if x==0 else None)
        ax.annotate(f'{mean:.2f}% ± {std:.2f} pp',(x-.24,mean+std),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
        ax.annotate(f'论文 {ref:.1f}%',(x+.24,ref),xytext=(8,-12),textcoords='offset points',fontsize=9,color=gray)
    ax.annotate('seed=0：21.6%',(1-.06,21.6),xytext=(.48,13),arrowprops={'arrowstyle':'-','color':gray},fontsize=9)
    ax.set(xlim=(-.65,1.65),ylim=(0,100),xticks=[0,1],xticklabels=['GCN（n=10）','MLP（n=10）'],ylabel='测试准确率（%）',title='固定公开划分：全部种子与均值 ± 样本标准差')
    ax.legend(loc='lower left',frameon=False,fontsize=8.5);grid(ax)
    finish(fig,'01_test_accuracy',['data/test_runs.csv','paper p7 Table 2/3'],'误差线为 sample std(ddof=1)，不是置信区间；全部点保留。')
    fig,axes=plt.subplots(2,1,figsize=(7.6,5.5),layout='constrained',sharex=True)
    seeds=np.arange(10)
    axes[0].scatter(seeds-.09,[scores[('gcn',s)] for s in seeds],color=blue,marker='o',s=40,label='GCN')
    axes[0].scatter(seeds+.09,[scores[('mlp',s)] for s in seeds],color=orange,marker='s',s=40,label='MLP')
    axes[0].set(ylabel='测试准确率（%）',ylim=(0,100),title='同种子的测试表现与配对差值');axes[0].legend(frameon=False,ncol=2,loc='lower right')
    differences=np.array([scores[('gcn',s)]-scores[('mlp',s)] for s in seeds])
    bars=axes[1].bar(seeds,differences,color=blue,width=.65)
    bars[0].set_facecolor('white');bars[0].set_edgecolor(blue);bars[0].set_hatch('///')
    axes[1].axhline(differences.mean(),color=gray,ls='--',lw=1,label=f'配对均值 {differences.mean():.2f} pp')
    for x,value in enumerate(differences):
        axes[1].text(x,value+1,f'{value:.1f}',ha='center',fontsize=8.5,
                     bbox={'facecolor':'white','edgecolor':'none','pad':.35},zorder=5)
    axes[1].set(ylabel='GCN − MLP（百分点）',ylim=(0,66),xlabel='预先固定的随机种子',xticks=seeds,xlim=(-.65,9.65));axes[1].legend(frameon=False,loc='upper right',fontsize=9)
    for ax in axes:grid(ax)
    finish(fig,'02_paired_seeds',['data/test_runs.csv','data/paired_differences.csv'],'种子为离散标识，散点不连线；所有 10 对均显示。')
    metrics={}
    for kind in ['gcn','mlp']:
        metric_path=ROOT/'results'/'metrics'/f'{kind}_seed00.csv'
        metrics[kind]=[{k:float(v) for k,v in r.items()} for r in read_csv(metric_path)]
        shutil.copyfile(metric_path,out/'data'/f'{kind}_seed00_metrics.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.6,5.5),layout='constrained',sharex=True)
    for row,kind in enumerate(['gcn','mlp']):
        data=metrics[kind];epochs=[r['epoch'] for r in data]
        for key,color,style,label in [('eval_train',blue,'-','训练（eval）'),('val',orange,'--','验证')]:
            axes[row,0].plot(epochs,[r[key+'_objective'] for r in data],c=color,ls=style,lw=1.7,label=label)
            axes[row,1].plot(epochs,[100*r[key+'_accuracy'] for r in data],c=color,ls=style,lw=1.7,label=label)
        for col in range(2):
            axes[row,col].axvline(epochs[-1],color=gray,ls=':',lw=1)
            axes[row,col].set_title(f'{kind.upper()} · seed=0 · 停于 {int(epochs[-1])} 轮',fontsize=10)
            axes[row,col].set_xlim(0,205);grid(axes[row,col])
        axes[row,0].set(ylabel='CE + 第一层 L2',ylim=(0,2.1))
        axes[row,1].set(ylabel='准确率（%）',ylim=(0,100))
    axes[0,0].legend(frameon=False,fontsize=8.5)
    for ax in axes[1]:ax.set_xlabel('Epoch')
    finish(fig,'03_seed0_training',['data/gcn_seed00_metrics.csv','data/mlp_seed00_metrics.csv'],'预先指定 seed=0；同一 0–200 epoch 横轴，MLP 不延长或补齐。')
    data=metrics['mlp'];subset=[r for r in data if r['epoch']>=17];x=[r['epoch'] for r in subset]
    previous=[np.mean([r['val_objective'] for r in data[int(e)-11:int(e)-1]]) for e in x]
    fig,axes=plt.subplots(3,1,figsize=(7.6,6.0),layout='constrained',sharex=True)
    axes[0].plot(x,[r['val_ce'] for r in subset],color=blue,marker='o',lw=1.7);axes[0].set(ylabel='验证 CE',title='MLP seed=0：末 6 轮的早停判据分解')
    axes[1].plot(x,[r['val_l2'] for r in subset],color=orange,marker='s',lw=1.7);axes[1].set_ylabel('第一层 L2')
    axes[2].plot(x,[r['val_objective'] for r in subset],color=blue,marker='o',label='当前验证 CE + L2')
    axes[2].plot(x,previous,color=orange,ls='--',marker='s',label='前 10 轮均值')
    axes[2].set(ylabel='验证目标（局部放大）',xlabel='Epoch',xticks=x,xlim=(16.65,22.35))
    axes[2].legend(frameon=False,fontsize=8.5,loc='upper right')
    for ax in axes:
        grid(ax);ax.ticklabel_format(style='plain',axis='y',useOffset=False)
    finish(fig,'04_mlp0_early_stop',['data/mlp_seed00_metrics.csv'],'各行独立纵轴；第 22 轮目标 1.93950760 > 前 10 轮均值 1.93944826。')
    fig,ax=plt.subplots(figsize=(7.6,3.2),layout='constrained')
    for offset,kind,color,hatch in [(-.18,'gcn',blue,None),(.18,'mlp',orange,'//')]:
        values=[int(next(r['selected_epoch'] for r in rows if r['model']==kind and int(r['seed'])==s)) for s in seeds]
        ax.bar(seeds+offset,values,width=.34,color=color,hatch=hatch,label=kind.upper())
    ax.set(xlabel='随机种子',ylabel='停止 Epoch',xticks=seeds,ylim=(0,225),title='全部 20 次运行的停止位置')
    ax.legend(ncol=2,frameon=False,loc='upper right');grid(ax)
    finish(fig,'05_stop_epochs',['data/test_runs.csv'],'上限 200；GCN seed=9 停于 192，MLP seed=0/1/3 停于 22/194/193。')
    write={'plotter_sha256':sha256(Path(__file__)),'inputs':{name:sha256(out/'data'/name) for name in [p.name for p in (out/'data').glob('*')]},'figures':assets,
        'paper_references':{'gcn':81.5,'mlp':55.1,'source':'arXiv:1609.02907v4, p7, Tables 2/3; 100 initializations'}}
    (out/'figure_manifest.json').write_text(json.dumps(write,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Generated 5 PNG + 5 SVG figures; source CSV and plotter snapshot saved.')


if __name__=='__main__':main()
