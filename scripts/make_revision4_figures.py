"""Scientific figures from the frozen information order and coefficient outputs."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
                    'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none',
                    'pdf.fonttype':42,'savefig.facecolor':'white'})

def save(fig,name):
    paths=[]
    for ext in ['png','pdf','svg']:
        path=OUT/(name+'.'+ext)
        fig.savefig(path,dpi=900 if ext=='png' else None,bbox_inches='tight',pad_inches=.04)
        paths.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    plt.close(fig)
    return paths

def timing():
    fig,ax=plt.subplots(figsize=(6.7,1.55))
    fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    boxes=[(.015,.245,'Event $u$','Store sensor context\nRequest reference'),
           (.357,.287,'Clock $u+d$','Predict current output\nThen release reference $y_u$'),
           (.740,.245,'Clock $u+d+1$','First output affected\nby the delivered reference')]
    for x,width,title,body in boxes:
        ax.add_patch(FancyBboxPatch((x,.22),width,.55,boxstyle='round,pad=0.008',
                                   facecolor='#f4f6f8',edgecolor='#4e5964',linewidth=.8))
        ax.text(x+width/2,.64,title,ha='center',va='center',fontsize=10)
        ax.text(x+width/2,.405,body,ha='center',va='center',fontsize=8.2,linespacing=1.6)
    for x1,x2 in [(.270,.347),(.654,.730)]:
        ax.annotate('',xy=(x2,.49),xytext=(x1,.49),arrowprops={'arrowstyle':'->','color':'#333333','lw':1.0})
    ax.text(.307,.87,'Delivery delay $d$',ha='center',va='center',fontsize=8.5)
    ax.text(.690,.105,'Update event state\nand propagate',ha='center',va='center',fontsize=8.2,linespacing=1.5)
    return save(fig,'revision4_information_timing')

def coefficients():
    source=ROOT/'results/revision4/components/weekly_coefficients_uncertainty.csv'
    frame=pd.read_csv(source)
    frame=frame[frame.family.eq('Zep') & ~frame.deployment.eq('Manchester__Zep344__s2')]
    segments=['Manchester__Zep188__s1','Manchester__Zep344__s1','London__Zep311__s2','York__Zep309__s1']
    assert set(frame.deployment.unique())==set(segments)
    labels=['Manchester Zep188','Manchester Zep344','London Zep311','York Zep309']
    colors=['#0072b2','#d55e00','#009e73','#cc79a7']
    styles=['-','--','-.',':']
    coefficients=[('baseline_ppb','Intercept (ppb)'),('pollutant_gain','NO$_2$ gain (ppb / ppb)'),
                  ('temperature_ppb_per_C','Temperature slope (ppb / °C)'),
                  ('humidity_ppb_per_percent','Humidity slope (ppb / %RH)')]
    fig,axes=plt.subplots(2,2,figsize=(6.7,4.35),sharex=True)
    for ax,(key,ylabel),panel in zip(axes.flat,coefficients,'abcd'):
        for segment,label,color,style in zip(segments,labels,colors,styles):
            d=frame[frame.deployment.eq(segment)&frame.coefficient.eq(key)].sort_values('elapsed_days')
            assert len(d)>1 and d[['elapsed_days','mean']].notna().all().all()
            ax.plot(d.elapsed_days,d['mean'],label=label,color=color,ls=style,lw=1.05)
        ax.set_ylabel(ylabel)
        ax.text(.02,.96,'('+panel+')',transform=ax.transAxes,va='top',fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#e4e4e4',lw=.5)
        ax.set_xlim(left=0)
    for ax in axes[1]:ax.set_xlabel('Elapsed days')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,
               frameon=False,bbox_to_anchor=(.54,1.0),fontsize=8.5)
    fig.subplots_adjust(left=.11,right=.985,bottom=.12,top=.85,wspace=.32,hspace=.26)
    receipt=dict(source=source.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 segments=segments,coefficient_panels=[x[0] for x in coefficients],
                 plotted_weekly_values=len(frame),uncertainty='State means only; no physical confidence intervals.')
    return save(fig,'revision4_coefficient_trajectories'),receipt

if __name__=='__main__':
    paths=timing();curves,receipt=coefficients();paths+=curves
    receipt.update(status='PASS',genuine_raster_dpi=900,vector_outputs=['pdf','svg'],outputs=paths)
    (ROOT/'evidence/revision4_figure_validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='outputs'},indent=2))
