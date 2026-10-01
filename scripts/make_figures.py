"""Scientific figures from completed audited result tables only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
    'svg.fonttype':'none','font.family':'DejaVu Sans'})
COLORS=['#175E89','#BF5A36','#55804E','#8C67A2','#8A6A38','#555555']
LABELS=dict(periodic='Dynamic AR + periodic',noAR='Dynamic white residual',hgb_stream='Immediate boosting',
    gp_stream='Immediate Gaussian process',ridge_stream='Immediate Ridge',ewrls='Recursive least squares',
    bias_ewma='Initial Ridge + bias update',risk='Risk, variable duration',risk_fixed='Risk, fixed duration',
    information_gain='Information gain',window_uncertainty='Uncertainty window',uncertainty='Pointwise uncertainty')


def save(fig,name):
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=900,bbox_inches='tight')
    plt.close(fig)


def main():
    a=pd.read_csv(ROOT/'results/tables/aggregate_metrics.csv');a=a[a.reference.eq('primary')]
    fig,axs=plt.subplots(1,2,figsize=(8.0,4.6))
    groups=[['periodic','ewrls','bias_ewma','hgb_stream','gp_stream','ridge_stream'],
        ['periodic','risk','risk_fixed','information_gain','window_uncertainty','uncertainty']]
    for ax,methods,title in zip(axs,groups,['(a) Correction models, common periodic mask','(b) Acquisition, common dynamic AR model']):
        for color,m in zip(COLORS,methods):
            d=a[a.method.eq(m)].sort_values('budget');assert len(d)==3,m
            ax.plot(d.budget,d.macro_instrument_rmse,'o-',color=color,label=LABELS[m],ms=3,lw=1.4)
        ax.set_xticks([12,24,48]);ax.set_xlabel('Additional attempted hours per 672-hour block')
        ax.set_ylabel('Mean physical-instrument RMSE (ppb)');ax.set_title(title,fontsize=9)
        ax.legend(frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.23));ax.grid(axis='y',alpha=.15)
    fig.tight_layout();save(fig,'figure1_error_budget')

    d=pd.read_csv(ROOT/'results/tables/instrument_metrics.csv')
    d=d[d.reference.eq('primary')&d.budget.eq(48)]
    p=d.pivot(index='instrument',columns='method',values='rmse')
    diff=(p.periodic-p.hgb_stream).sort_values()
    fig,ax=plt.subplots(figsize=(6.5,4.4))
    ax.barh(diff.index,diff.values,color=np.where(diff<0,'#175E89','#BF5A36'))
    ax.axvline(0,color='black',lw=.7);ax.set_xlabel('Dynamic AR minus immediate boosting RMSE (ppb)')
    ax.set_ylabel('Physical instrument');ax.grid(axis='x',alpha=.15)
    fig.tight_layout();save(fig,'figure2_instrument_errors')

    d=pd.read_csv(ROOT/'results/tables/monthly_metrics.csv')
    d=d[d.reference.eq('primary')&d.budget.eq(48)&d.family.eq('Zep')&d.instrument_hours.ge(168)]
    fig,axs=plt.subplots(2,2,figsize=(8.,4.8),sharex=True)
    for ax,inst in zip(axs.flat,['Zep188','Zep344','Zep311','Zep309']):
        for color,m in zip(COLORS,['periodic','hgb_stream','bias_ewma','risk_fixed']):
            z=d[d.instrument.eq(inst)&d.method.eq(m)].sort_values('period')
            ax.plot(pd.to_datetime(z.period),z.rmse,color=color,lw=1.1,label=LABELS[m])
        ax.set_title(inst,fontsize=9);ax.set_ylabel('Monthly RMSE (ppb)');ax.grid(axis='y',alpha=.15)
        ax.tick_params(axis='x',rotation=25,labelsize=7)
    axs[0,0].legend(frameon=False,fontsize=6.5);fig.tight_layout();save(fig,'figure3_long_records')

    d=pd.read_csv(ROOT/'results/uncertainty_diagnostics.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.,3.3))
    for j,m in enumerate(['periodic','risk','risk_fixed']):
        z=d[d.method.eq(m)]
        offset=(np.arange(len(z))-(len(z)-1)/2)*.017
        axs[0].scatter(np.full(len(z),j)+offset,100*z.nominal95_coverage,s=15,color=COLORS[j],alpha=.6)
        axs[0].scatter(j,100*z.nominal95_coverage.mean(),marker='D',s=38,color=COLORS[j])
        axs[1].scatter(z.mean_gaussian_variance,z.observed_mse,s=20,color=COLORS[j],label=LABELS[m],alpha=.6)
    axs[0].axhline(95,color='black',ls='--',lw=.8,label='Nominal 95%')
    axs[0].set_xticks(range(3),['Periodic','Variable risk','Fixed risk']);axs[0].set_ylabel('Observed interval coverage (%)')
    axs[0].set_title('(a) Instrument coverage and its mean',fontsize=9)
    vmin=min(d.observed_mse.min(),d.mean_gaussian_variance.min())*.8
    vmax=max(d.observed_mse.max(),d.mean_gaussian_variance.max())*1.1
    axs[1].plot([vmin,vmax],[vmin,vmax],ls='--',color='black',lw=.8)
    axs[1].set_xscale('log');axs[1].set_yscale('log');axs[1].set_xlabel('Mean Gaussian variance (ppb²)')
    axs[1].set_ylabel('Observed squared error (ppb²)');axs[1].set_title('(b) Model variance and realized error',fontsize=9)
    axs[1].legend(frameon=False,fontsize=6.5);fig.tight_layout();save(fig,'figure4_uncertainty')
    print('Saved four SVG and 900-dpi PNG figures')


if __name__=='__main__':main()
