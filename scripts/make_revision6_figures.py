"""Draw service-condition curves and device confidence intervals at print size."""
from pathlib import Path
import json,shutil,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import make_nature_figures as style
from verify_nature_figures import check_geometry
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results/revision6'

def service(width):
    d=pd.read_csv(R/'fixed_delay_corrected/matrix_aggregate_metrics.csv')
    fig=style.canvas(width,72)
    axes=[fig.add_axes([.09,.24,.38,.56]),fig.add_axes([.59,.24,.38,.56])]
    colors=[style.P['grey'],style.P['orange'],style.P['blue']]
    markers=['s','^','o']
    for ax,sharing,letter,title in zip(axes,[True,False],'ab',['Sharing enabled','Sharing disabled']):
        for method,label,color,marker in zip(['periodic','original','aware_retained'],
                ['Periodic','Original objective','Delivery-aware'],colors,markers):
            g=d[(d.shared==sharing)&(d.method==method)].sort_values('delay')
            ax.plot(g.delay,g.macro_instrument_rmse,marker=marker,color=color,label=label,lw=1.1,ms=3.5)
        ax.set_xticks([0,24,72,168]);ax.set_xlim(-6,174);ax.set_ylim(4.05,6.0)
        ax.set_yticks([4.2,4.6,5.0,5.4,5.8]);ax.set_xlabel('Reference delivery delay (h)',labelpad=5)
        ax.set_ylabel('Instrument-equal RMSE (ppb)',labelpad=4)
        style.clean_axis(ax);style.panel(fig,ax,letter,title)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.53,.01))
    return fig

def devices(width):
    d=pd.read_csv(R/'statistics/device_contrasts_full_support_7d.csv').sort_values('difference').reset_index(drop=True)
    fig=style.canvas(width,112)
    ax=fig.add_axes([.25,.17,.71,.73])
    ax.axvline(0,color=style.P['grey'],lw=.6,ls=(0,(3,3)),zorder=0)
    for i,row in d.iterrows():
        extended=row['instrument'] in ['Zep188','Zep309','Zep311','Zep344']
        color=style.P['blue'] if extended else style.P['grey']
        ax.plot([row.ci_low,row.ci_high],[i,i],color=color,lw=.9)
        ax.plot(row.difference,i,'o' if extended else 's',ms=3.5,color=color)
    ax.set_yticks(range(len(d)),[f"{r.instrument}  ·  {r.site}" for r in d.itertuples()])
    ax.invert_yaxis();ax.set_ylim(17.8,-.8)
    lo=min(-4.8,d.ci_low.min()-.15);hi=max(2.0,d.ci_high.max()+.15)
    ax.set_xlim(lo,hi);ax.set_xlabel('Fixed-risk − periodic RMSE (ppb)',labelpad=5)
    style.clean_axis(ax);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.text(.25,.945,'Observed instrument contrasts',fontsize=8,weight='bold')
    handles=[style.Line2D([],[],marker='o',lw=0,color=style.P['blue'],label='Extended records'),
             style.Line2D([],[],marker='s',lw=0,color=style.P['grey'],label='Shorter records')]
    fig.legend(handles=handles,loc='lower center',ncol=2,bbox_to_anchor=(.58,.01))
    return fig

def main():
    receipts=[]
    for variant in ['word','latex']:
        out=ROOT/f'results/figures_revision6_{variant}';out.mkdir(parents=True,exist_ok=True)
        for file in (ROOT/f'results/figures_nature_{variant}').iterdir():
            if file.is_file():shutil.copy2(file,out/file.name)
        width=175.26 if variant=='word' else 181
        for name,builder in [('revision6_figure2_service_latency',service),('revision6_figure3_instrument_intervals',devices)]:
            fig=builder(width);geometry=check_geometry(fig)
            assert not geometry['text_overlaps'],geometry
            assert not geometry['text_outside_canvas'],geometry
            for ax in fig.axes:
                for line in ax.lines:
                    if name.endswith('service_latency'):
                        assert np.all((np.asarray(line.get_ydata())>=ax.get_ylim()[0])&(np.asarray(line.get_ydata())<=ax.get_ylim()[1])), 'data clipped'
            for ext in ['pdf','svg','png']:
                file=out/f'{name}.{ext}';fig.savefig(file,dpi=1200 if ext=='png' else 180)
            preview=ROOT/f'temp/revision6_previews/{variant}';preview.mkdir(parents=True,exist_ok=True)
            fig.savefig(preview/f'{name}.png',dpi=180)
            receipts.append(dict(variant=variant,name=name,width_mm=width,height_mm=fig.get_size_inches()[1]/style.MM,
                geometry=geometry,pdf_sha256=hashlib.sha256((out/f'{name}.pdf').read_bytes()).hexdigest()))
            plt.close(fig)
    (ROOT/'evidence/revision6/figure_geometry.json').write_text(json.dumps(receipts,indent=2))
    print('PASS',len(receipts),'print-size vector/raster figures, no text conflict')

if __name__=='__main__':main()
