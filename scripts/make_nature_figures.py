"""Redraw the six frozen-data figures at their actual insertion sizes.

Nature figure-guide typography, accessible colour and evidence-first layouts;
IEEE column dimensions are retained. No model fitting or statistics run here.
"""
from pathlib import Path
import os, json, hashlib, argparse
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'temp/matplotlib_nature'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.ticker import MaxNLocator, FuncFormatter
import pandas as pd
import numpy as np

MM = 1 / 25.4
P = {'blue': '#0072B2', 'orange': '#D55E00', 'green': '#009E73',
     'purple': '#CC79A7', 'grey': '#646464', 'ink': '#202020',
     'light': '#D5DADD', 'cyan': '#56B4E9'}
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'text.color': P['ink'], 'axes.labelcolor': P['ink'],
    'xtick.color': P['ink'], 'ytick.color': P['ink'],
    'axes.linewidth': .5, 'axes.edgecolor': P['ink'],
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': .5, 'ytick.major.width': .5,
    'lines.linewidth': 1, 'lines.markersize': 3,
    'legend.frameon': False, 'legend.handlelength': 2.1,
    'legend.handletextpad': .6, 'legend.columnspacing': 1.7,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'text.usetex': False, 'mathtext.fontset': 'dejavusans',
    'savefig.bbox': None, 'savefig.facecolor': 'white',
    'figure.facecolor': 'white', 'figure.dpi': 180,
})
DATA_SOURCES = set()

def read(relative):
    path = ROOT / relative
    DATA_SOURCES.add(path)
    return pd.read_csv(path)

def canvas(width, height):
    return plt.figure(figsize=(width * MM, height * MM))

def panel(fig, ax, letter, title=None):
    b = ax.get_position()
    fig.text(b.x0, b.y1 + .035, letter, fontsize=8, weight='bold', va='bottom')
    if title:
        fig.text(b.x0 + .025, b.y1 + .035, title, fontsize=7, va='bottom')

def clean_axis(ax):
    ax.grid(False)
    ax.tick_params(direction='out', pad=3)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def schematic(width, height):
    fig = canvas(width, height)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width); ax.set_ylim(0, height); ax.axis('off')
    return fig, ax

def arrow(ax, start, end, color=None, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=6,
        lw=.75, color=color or P['grey'], linestyle='--' if dashed else '-'))

def card(ax, x, y, w, h, title, body, accent=None):
    ax.add_patch(Rectangle((x, y), w, h, fc='white', ec=P['light'], lw=.6))
    ax.plot([x, x], [y, y+h], lw=2, color=accent or P['blue'], solid_capstyle='butt')
    ax.text(x+3, y+h-4, title, ha='left', va='top', fontsize=7, weight='bold')
    ax.text(x+3, y+h-10, body, ha='left', va='top', fontsize=7, linespacing=1.35)

def information_timing(width):
    if width < 100:
        fig, ax = schematic(width, 86)
        events = [(60, '1   Event u', 'Store sensor context\nRequest reference yᵤ'),
                  (34, '2   Clock u + d', 'Issue the current output\nThen release reference yᵤ'),
                  (8, '3   Clock u + d + 1', 'First output informed by yᵤ\nPropagate the updated event state')]
        for y, title, body in events:
            card(ax, 6, y, width-12, 21, title, body)
        arrow(ax, (width/2, 60), (width/2, 55))
        arrow(ax, (width/2, 34), (width/2, 29))
    else:
        fig, ax = schematic(width, 48)
        w = (width-24)/3
        for i, (title, body) in enumerate([
            ('1   Event u', 'Store sensor context\nRequest reference yᵤ'),
            ('2   Clock u + d', 'Issue the current output\nThen release reference yᵤ'),
            ('3   Clock u + d + 1', 'First output informed by yᵤ\nPropagate the updated event state')]):
            card(ax, 5+i*(w+7), 12, w, 24, title, body)
        for i in range(2):
            arrow(ax, (5+(i+1)*w+i*7+.5, 24), (5+(i+1)*(w+7)-.5, 24))
        ax.text(5+w/2, 42, 'Reference event', va='center', ha='center', fontsize=7)
        ax.text(5+(w+7)+w/2, 42, 'Delivery delay d', va='center', ha='center', fontsize=7)
        ax.text(5+2*(w+7)+w/2, 42, 'First affected output', va='center', ha='center', fontsize=7)
    return fig

def coefficient_trajectories(width):
    d = read('results/revision4/components/weekly_coefficients_uncertainty.csv')
    d = d[d.family.eq('Zep') & ~d.deployment.eq('Manchester__Zep344__s2')]
    segments = ['Manchester__Zep188__s1','Manchester__Zep344__s1',
                'London__Zep311__s2','York__Zep309__s1']
    labels = ['Manchester Zep188','Manchester Zep344','London Zep311','York Zep309']
    colors = [P['blue'],P['orange'],P['green'],P['purple']]
    styles = ['-',(0,(4,1.5)),(0,(3,1,1,1)),(0,(1,1.5))]
    fields = [('baseline_ppb','Intercept (ppb)'),
              ('pollutant_gain','NO₂ gain (ppb / ppb)'),
              ('temperature_ppb_per_C','Temperature slope (ppb / °C)'),
              ('humidity_ppb_per_percent','Humidity slope (ppb / %RH)')]
    fig = canvas(width, 119)
    axes = [fig.add_axes(p) for p in [[.105,.565,.365,.285],[.605,.565,.365,.285],
                                     [.105,.14,.365,.285],[.605,.14,.365,.285]]]
    for ax, (field,label), letter in zip(axes,fields,'abcd'):
        for segment,name,color,style in zip(segments,labels,colors,styles):
            g=d[d.deployment.eq(segment)&d.coefficient.eq(field)].sort_values('elapsed_days')
            assert len(g)>1 and g[['elapsed_days','mean']].notna().all().all()
            ax.plot(g.elapsed_days,g['mean'],color=color,ls=style,lw=.85,label=name)
        label=label.replace('NO₂','NO$_2$')
        ax.set_xlim(0,540); ax.set_xticks([0,180,360,540]); ax.set_ylabel(label, labelpad=5)
        clean_axis(ax); panel(fig,ax,letter)
        if field=='pollutant_gain': ax.set_ylim(bottom=0)
        if field=='temperature_ppb_per_C': ax.set_yticks([-1.5,-1,-.5,0])
    for ax in axes[2:]: ax.set_xlabel('Elapsed days', labelpad=5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2,
               bbox_to_anchor=(.53,.998), borderaxespad=0, labelspacing=.7)
    return fig

def instrument_contrasts(width):
    d=read('results/revision/diagnostics/B48_fixed_vs_periodic_instruments.csv')
    d['long']=d.evidence_group.eq('long_Zep')
    d=d.sort_values(['long','site','instrument'],ascending=[False,True,True]).reset_index(drop=True)
    fig=canvas(width,139)
    ax=fig.add_axes([.235,.155,.56,.745])
    yy=np.arange(len(d)); ax.set_ylim(17.7,-.7); ax.set_xlim(-2.5,1.25)
    for i,r in enumerate(d.itertuples()):
        color=P['blue'] if r.long else P['grey']
        ax.plot([0,r.rmse_difference],[i,i],color=color,lw=.75,alpha=.65)
        ax.plot(r.rmse_difference,i,'o' if r.long else 's',ms=4.2,
                mfc=color if r.long else 'white',mec=color,mew=.8)
    ax.axvline(0,color=P['ink'],lw=.65,zorder=0)
    ax.axhline(3.5,color=P['light'],lw=.55,zorder=0)
    ax.set_yticks(yy,[f'{r.site[:3]} / {r.instrument}' for r in d.itertuples()])
    ax.tick_params(axis='y',length=0,pad=6); ax.spines['left'].set_visible(False)
    ax.set_xticks([-2,-1,0,1]); ax.set_xlabel('Fixed-risk minus periodic AR RMSE (ppb)',labelpad=7)
    clean_axis(ax)
    for i,r in enumerate(d.itertuples()):
        y=fig.transFigure.inverted().transform(ax.transData.transform((0,i)))[1]
        fig.text(.885,y,f'{r.instrument_hours_fixed:,}',ha='right',va='center',fontsize=7)
        fig.text(.975,y,f'{r.rmse_difference:+.3f}',ha='right',va='center',fontsize=7)
    fig.text(.885,.927,'Scored h',ha='right',va='bottom',fontsize=7)
    fig.text(.975,.927,'ΔRMSE',ha='right',va='bottom',fontsize=7)
    fig.text(.235,.927,'Fixed risk improves',ha='left',va='bottom',fontsize=7)
    fig.text(.795,.927,'Periodic improves',ha='right',va='bottom',fontsize=7)
    handles=[Line2D([],[],marker='o',ls='',color=P['blue'],ms=4,label='Extended record (4)'),
             Line2D([],[],marker='s',ls='',mfc='white',mec=P['grey'],ms=4,label='Shorter product record (14)')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.56,.018),ncol=2)
    return fig

def local_window_value(width):
    p=read('results/revision/branch/all_points.csv')
    s=read('results/revision/branch/summary.csv')
    names=['Zep188','Zep309','Zep311','Zep344']; colors=[P['blue'],P['orange'],P['green'],P['purple']]
    fig=canvas(width,93)
    ax=fig.add_axes([.095,.23,.49,.59]); bx=fig.add_axes([.72,.23,.255,.59])
    for name,color,marker in zip(names,colors,['o','s','^','D']):
        g=p[p.instrument.eq(name)]
        ax.scatter(g.predicted_difference,g.realized_mse_difference,s=11,alpha=.72,
                   color=color,marker=marker,linewidths=.25,edgecolors='white',label=f'{name} (n = {len(g)})')
    ax.set_xscale('symlog',linthresh=.5); ax.set_yscale('symlog',linthresh=5)
    ax.set_xticks([-1,0,1,10,100],['−1','0','1','10','100'])
    ax.set_yticks([-100,-10,0,10,100,1000],['−100','−10','0','10','100','1000'])
    ax.axvline(0,lw=.5,color=P['light']); ax.axhline(0,lw=.5,color=P['light'])
    ax.set_xlabel('Predicted immediate-window\nadvantage (ppb²)',labelpad=6)
    ax.set_ylabel('Realized immediate-window\nadvantage (ppb²)',labelpad=6)
    rows=s[s.instrument.ne('pooled')].set_index('instrument').loc[names]
    for i,(name,color) in enumerate(zip(names,colors)):
        v=rows.loc[name,'spearman_mse']; bx.plot([0,v],[i,i],color=color,lw=1)
        bx.plot(v,i,'o',mfc=color,mec='white',mew=.4,ms=4.4)
    bx.set_yticks(range(4),names); bx.set_ylim(3.8,-.8); bx.set_xlim(-.2,.4)
    bx.set_xticks([-.2,0,.2,.4]); bx.axvline(0,lw=.5,color=P['light'])
    bx.set_xlabel('Spearman correlation',labelpad=6); bx.spines['left'].set_visible(False)
    bx.tick_params(axis='y',length=0)
    for a in [ax,bx]: clean_axis(a)
    panel(fig,ax,'a'); panel(fig,bx,'b')
    pooled=s[s.instrument.eq('pooled')].iloc[0]
    fig.text(.72,.96,f'Pooled ρ = {pooled.spearman_mse:.4f}',ha='left',va='top',fontsize=7)
    fig.text(.72,.913,f'Sign agreement = {100*pooled.sign_agreement:.1f}%',ha='left',va='top',fontsize=7)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=4,
               bbox_to_anchor=(.53,.015),columnspacing=1.3,handletextpad=.35)
    return fig

def access_tradeoff(width):
    c=read('results/revision/diagnostics/error_total_access_curves.csv')
    f=read('results/revision/diagnostics/minimum_tested_budget_frontier.csv')
    methods=[('periodic','AR periodic',P['blue'],'o','-'),
             ('risk_fixed','AR fixed risk',P['orange'],'s','-'),
             ('risk','AR variable risk',P['green'],'^','-'),
             ('gp_stream','GP periodic',P['purple'],'D','--'),
             ('hgb_stream','Boosting periodic',P['grey'],'v','--'),
             ('uncertainty','AR uncertainty',P['cyan'],'P',':')]
    fig=canvas(width,90)
    axes=[fig.add_axes(pos) for pos in [[.075,.31,.22,.47],[.415,.31,.22,.47],[.75,.31,.22,.47]]]
    for method,label,color,marker,style in methods:
        g=c[c.method.eq(method)&c.lambda_hours.eq(0)].sort_values('budget')
        axes[0].plot(g.total_attempted_hours,g.macro_instrument_rmse,color=color,marker=marker,
                     ms=3.4,mew=.4,mec='white',ls=style,label=label)
        for ax,lam in zip(axes[1:],[0,72]):
            q=f[f.method.eq(method)&f.lambda_hours.eq(lam)&f.endpoint.eq('macro_instrument_rmse')].sort_values('target_ppb')
            ax.step(q.target_ppb,q.equivalent_hours,where='post',color=color,lw=1,ls=style)
    axes[0].set_xlabel('Total reference\naccess (h)',labelpad=6); axes[0].set_ylabel('Mean instrument RMSE (ppb)',labelpad=5)
    axes[0].set_xticks([2500,3250,4000],['2,500','3,250','4,000'])
    axes[0].yaxis.set_major_locator(MaxNLocator(4))
    for ax in axes[1:]:
        ax.set_xlabel('RMSE requirement\n(ppb)',labelpad=6); ax.set_ylabel('Minimum tested\nreference access (h)',labelpad=4)
        ax.set_xlim(3,8); ax.set_xticks([3,5,8]); ax.set_ylim(0,20000)
        ax.set_yticks([0,10000,20000],['0','10,000','20,000'])
    axes[1].set_ylim(0,5000)
    axes[1].set_yticks([0,2500,5000],['0','2,500','5,000'])
    for ax,letter,title in zip(axes,'abc',['Original budgets','λ = 0 h','λ = 72 h']):
        clean_axis(ax); panel(fig,ax,letter,title)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,
               bbox_to_anchor=(.53,.035),labelspacing=.8)
    return fig

def service_procedure(width):
    if width < 100:
        fig,ax=schematic(width,119)
        card(ax,6,91,width-12,23,'1   Establish measurement support',
             'Compatible colocated reference\nand timestamped sensor records?')
        ax.text(9,83,'No: evaluate deployment\nsupport first',fontsize=7,va='top',linespacing=1.35)
        arrow(ax,(25,91),(25,86))
        arrow(ax,(width-12,91),(width-12,68)); ax.text(width-9,73,'Yes',fontsize=7)
        card(ax,6,46,width-12,22,'2   Specify the information contract',
             'Commissioning, latency and sharing\nQuality status and total access')
        arrow(ax,(width/2,46),(width/2,40))
        card(ax,6,19,width-12,21,'3   Compare correction and calendars',
             'Competitive periodic correction\nSelective and random access')
        arrow(ax,(width/2,19),(width/2,13))
        ax.text(9,9,'Select access only if the trade-off improves;\nvalidate on a reserved later field period.',
                fontsize=7,va='center',linespacing=1.35)
    else:
        fig,ax=schematic(width,89)
        w=(width-23)/2
        card(ax,5,58,w,25,'1   Establish measurement support',
             'Compatible colocated reference\nand timestamped sensor records?')
        card(ax,13+w,58,w,25,'2   Specify the information contract',
             'Commissioning, latency and sharing\nQuality status and total access')
        arrow(ax,(5+w,70),(13+w,70)); ax.text(9+w,76,'Yes',ha='center',fontsize=7)
        ax.text(8,47,'No: evaluate deployment support first',fontsize=7,va='center')
        arrow(ax,(13+1.5*w,58),(13+1.5*w,35))
        card(ax,13+w,10,w,25,'3   Compare correction and calendars',
             'Competitive periodic correction\nSelective and random access')
        card(ax,5,10,w,25,'4   Select and validate',
             'Select access if the trade-off improves\nPilot on a reserved later field period',accent=P['green'])
        arrow(ax,(13+w,22),(5+w,22))
    return fig

BUILDERS=[('revision5_figure1_information_timing',information_timing),
          ('revision5_figure2_coefficient_trajectories',coefficient_trajectories),
          ('revision5_figure3_instrument_contrasts',instrument_contrasts),
          ('revision5_figure4_local_window_value',local_window_value),
          ('revision5_figureS1_access_tradeoff',access_tradeoff),
          ('revision5_figureS3_service_procedure',service_procedure)]
SCHEMATICS={BUILDERS[0][0],BUILDERS[-1][0]}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--variant',choices=['word','latex'],default='latex')
    parser.add_argument('--dpi',type=int,default=1200); args=parser.parse_args()
    out=ROOT / ('results/figures_nature_word' if args.variant=='word' else 'results/figures_nature_latex')
    out.mkdir(parents=True,exist_ok=True); previews=ROOT/'temp/nature_previews'/args.variant
    previews.mkdir(parents=True,exist_ok=True)
    receipt={'variant':args.variant,'outputs':[],'font':'Arial','text_pt':7,'panel_pt':8,'raster_dpi':args.dpi}
    for name,builder in BUILDERS:
        width=175.26 if args.variant=='word' else (88 if name in SCHEMATICS else 181)
        suffix='_single' if args.variant=='latex' and name in SCHEMATICS else ''
        fig=builder(width)
        for ext in ['pdf','svg','png']:
            path=out/f'{name}{suffix}.{ext}'; fig.savefig(path,dpi=args.dpi if ext=='png' else 180)
            receipt['outputs'].append({'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'width_mm':width,
                'height_mm':float(fig.get_size_inches()[1]/MM)})
        fig.savefig(previews/f'{name}.png',dpi=180); plt.close(fig)
        print(args.variant,name,width,'mm')
    receipt['sources']=[{'path':s.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(s.read_bytes()).hexdigest()} for s in sorted(DATA_SOURCES)]
    (ROOT/f'evidence/nature_figures_{args.variant}.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')

if __name__=='__main__': main()
