"""Draw the explicitly proposed evidence-guided reference-service procedure."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(8.8,5.4));ax.set_xlim(0,11);ax.set_ylim(-.12,7.3);ax.axis('off')
    def box(x,y,w,height,text,fill='#f1f4f5'):
        ax.add_patch(FancyBboxPatch((x,y),w,height,boxstyle='round,pad=0.07,rounding_size=0.07',facecolor=fill,edgecolor='#465763',linewidth=.9))
        ax.text(x+w/2,y+height/2,text,ha='center',va='center',fontsize=10)
    def arrow(a,b,label=None,offset=(0,0)):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','color':'#465763','lw':1.1,'shrinkA':4,'shrinkB':4})
        if label:ax.text((a[0]+b[0])/2+offset[0],(a[1]+b[1])/2+offset[1],label,ha='center',va='center',fontsize=9,color='#233340')
    box(2.5,6.2,5.5,.82,'Compatible colocated reference\nand timestamped sensor records?')
    box(.1,4.65,2.4,1.05,'Field access or relocation\nrequires a separate\ndeployment evaluation',fill='#fff5ef')
    box(3.1,4.65,4.3,1.05,'Specify commissioning, label latency,\nsharing, quality and total access')
    box(3.1,3.05,4.3,1.05,'Benchmark strong periodic correction\nagainst selective and random calendars')
    box(3.1,1.48,4.3,1.05,'Selective access improves the required\nerror–access trade-off with acceptable\ndevice accuracy and availability?')
    box(8.1,1.45,2.65,1.1,'Retain best periodic service\nor revise the reference\ninformation contract',fill='#fff5ef')
    box(3.1,.04,4.3,.95,'Pilot the selective candidate\non a reserved later evaluation period')
    arrow((5.25,6.2),(5.25,5.7),'Yes',(.35,0))
    arrow((2.5,6.4),(1.3,5.7),'No',(-.25,.1))
    arrow((5.25,4.65),(5.25,4.1))
    arrow((5.25,3.05),(5.25,2.53))
    arrow((5.25,1.48),(5.25,.99),'Yes',(.35,0))
    arrow((7.4,2.0),(8.1,2.0),'No',(0,.2))
    fig.subplots_adjust(left=.012,right=.995,top=.995,bottom=.005)
    out=ROOT/'results/figures';out.mkdir(parents=True,exist_ok=True)
    for suffix,dpi in [('svg',None),('png',900)]:
        fig.savefig(out/f'revision2_figure4_decision.{suffix}',dpi=dpi,bbox_inches='tight',pad_inches=.04)
    plt.close(fig);print('Saved proposed decision procedure in SVG and 900-dpi PNG.')

if __name__=='__main__':main()
