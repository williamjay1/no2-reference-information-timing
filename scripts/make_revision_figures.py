"""Scientific plots from fixed saved revision tables; no model selection."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'temp/matplotlib'))
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D=ROOT/'results/revision/diagnostics';F=ROOT/'results/figures'
F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})


def save(fig,name):
    fig.savefig(F/(name+'.svg'),bbox_inches='tight')
    fig.savefig(F/(name+'.png'),dpi=900,bbox_inches='tight')
    plt.close(fig)


def instruments():
    x=pd.read_csv(D/'B48_fixed_vs_periodic_instruments.csv')
    x['long']=x.evidence_group.eq('long_Zep')
    x=x.sort_values(['long','site','instrument'],ascending=[False,True,True]).reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(8.4,5.3))
    colors=np.where(x.long,'#174a70','#8eaba8')
    ax.barh(np.arange(len(x)),x.rmse_difference,color=colors,height=.67)
    ax.axvline(0,color='black',lw=.9)
    ax.set_yticks(np.arange(len(x)),[f'{r.site[:3]} / {r.instrument}  ({r.instrument_hours_fixed:,} h)' for r in x.itertuples()])
    ax.invert_yaxis();ax.set_xlabel('Fixed-risk minus periodic AR RMSE (ppb)')
    ax.set_xlim(min(-2.5,x.rmse_difference.min()-.3),max(1.,x.rmse_difference.max()+.25))
    ax.axhline(3.5,color='#bbbbbb',ls=':',lw=1)
    ax.text(.02,.01,'Dark: four long records; pale: fourteen shorter product records',transform=ax.transAxes,fontsize=8)
    fig.tight_layout();save(fig,'revision_figure1_instruments')


def costs():
    curves=pd.read_csv(D/'error_total_access_curves.csv')
    front=pd.read_csv(D/'minimum_tested_budget_frontier.csv')
    labels={'periodic':'AR periodic','risk_fixed':'AR fixed risk','risk':'AR variable risk',
        'gp_stream':'GP periodic','hgb_stream':'Boosting periodic','uncertainty':'AR uncertainty'}
    colors=['#263d64','#e07634','#539798','#904b83','#627f34','#927753']
    fig,axs=plt.subplots(1,3,figsize=(10.2,3.3))
    for (method,label),color in zip(labels.items(),colors):
        c=curves[curves.method.eq(method)&curves.lambda_hours.eq(0)].sort_values('budget')
        axs[0].plot(c.total_attempted_hours,c.macro_instrument_rmse,'o-',ms=4,color=color,label=label)
        for ax,lam in zip(axs[1:],[0,72]):
            d=front[front.method.eq(method)&front.lambda_hours.eq(lam)&front.endpoint.eq('macro_instrument_rmse')].sort_values('target_ppb')
            # Unattained targets remain NaN; no extrapolated lower budget.
            ax.step(d.target_ppb,d.equivalent_hours,where='post',color=color)
    axs[0].set(xlabel='Setup + additional requested hours',ylabel='Mean instrument RMSE (ppb)',title='(a) Achieved error')
    for ax,lam,title in zip(axs[1:],[0,72],['(b) No activation overhead','(c) 72-hour activation scenario']):
        ax.set(xlabel='Analytical RMSE requirement (ppb)',ylabel='Minimum tested access cost (hours)',title=title,xlim=(3,8))
        ax.ticklabel_format(style='plain',axis='y')
    for ax in axs:ax.grid(axis='y',color='#dedede',lw=.5)
    handles,names=axs[0].get_legend_handles_labels()
    fig.legend(handles,names,loc='lower center',bbox_to_anchor=(.5,-.035),ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.12,1,1));save(fig,'revision_figure2_access')


def branches():
    data=pd.read_csv(ROOT/'results/revision/branch/all_points.csv')
    summary=pd.read_csv(ROOT/'results/revision/branch/summary.csv')
    fig,axs=plt.subplots(1,2,figsize=(8.4,3.8))
    colors=['#174a70','#e07634','#627f34','#904b83']
    for (name,g),color in zip(data.groupby('instrument'),colors):
        axs[0].scatter(g.predicted_difference,g.realized_mse_difference,s=15,alpha=.7,color=color,label=f'{name} (n={len(g)})')
    axs[0].axhline(0,color='#777777',lw=.7);axs[0].axvline(0,color='#777777',lw=.7)
    axs[0].set(xlabel='Predicted immediate-window advantage (ppb²)',ylabel='Realized immediate-window advantage (ppb²)',title='(a) All eligible weekly origins')
    axs[0].set_yscale('symlog',linthresh=5.)
    axs[0].legend(fontsize=8,frameon=False,loc='upper right')
    d=summary[summary.instrument.ne('pooled')]
    axs[1].bar(np.arange(len(d)),d.spearman_mse,color=colors)
    axs[1].axhline(0,color='#777777',lw=.7)
    axs[1].set(xticks=np.arange(len(d)),xticklabels=d.instrument,ylabel='Spearman correlation',title='(b) Ranking within each instrument',ylim=(-.4,.5))
    axs[1].text(.04,.93,'Pooled ρ = 0.0045; sign agreement = 52.4%',transform=axs[1].transAxes,fontsize=8)
    fig.tight_layout();save(fig,'revision_figure3_window_value')


if __name__=='__main__':
    instruments();costs();branches()
    print('Three revision figures exported as SVG and 900-dpi PNG.')
