"""Vector graphical abstract, generated directly from frozen article tables.

IEEE upload: 660 x 295 pixels, 300 dpi metadata, gagraphic filename.
PDF/SVG master: 3 inch width for the JSEN first-page abstract area.
"""
from pathlib import Path
import json, hashlib, os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'temp/matplotlib_nature'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from PIL import Image
from make_nature_figures import P

OUT=ROOT/'results/graphical_abstract'; OUT.mkdir(parents=True,exist_ok=True)
data=json.loads((ROOT/'manuscript/article_nature.json').read_text(encoding='utf-8'))
table2=next(b for b in data['blocks'] if b['type']=='table' and b['caption'].startswith('Table 2.'))
table3=next(b for b in data['blocks'] if b['type']=='table' and b['caption'].startswith('Table 3.'))
immediate=table2['rows'][0]
# Use the 72 hour delay-aware row by its explicit text, never a positional guess.
late=[r for r in table3['rows'] if '72' in str(r[0]) and 'development-selected' in str(r[0]).lower()]
assert len(late)==1,table3['rows']
late=late[0]
values=[float(immediate[2]),float(immediate[3]),float(late[1]),float(late[2])]
assert values==[4.582,4.161,5.060,4.664],values

fig=plt.figure(figsize=(3,295/220),dpi=220)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,76.2); ax.set_ylim(0,34.05909); ax.axis('off')
def text(x,y,s,size=6,**kw):
    return ax.text(x,y,s,fontsize=size,color=P['ink'],ha=kw.pop('ha','left'),
                   va=kw.pop('va','center'),linespacing=1.22,**kw)
def arrow(a,b,color=P['grey']):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=5,lw=.6,color=color))

# Distinct evidence roles; quiet separators rather than coloured background boxes.
for x in [25.3,51.5]: ax.plot([x,x],[3,31],color=P['light'],lw=.5)
text(2,30,'Public field data',weight='bold',size=6.2)
text(28,30,'Plan for delivery',weight='bold',size=6.2)
text(54,30,'Lower NO$_2$ error',weight='bold',size=6.2)

# Sensor and reference are functional symbols rather than decorative imagery.
ax.add_patch(Rectangle((3,17.5),6.2,8,fc='white',ec=P['ink'],lw=.6))
ax.add_patch(Rectangle((4.2,22.5),3.8,1.6,fc='white',ec=P['blue'],lw=.65))
for yy in [19,20.2]: ax.plot([4.2,8],[yy,yy],color=P['light'],lw=.55)
ax.add_patch(Rectangle((16,17.5),6.2,8,fc='white',ec=P['ink'],lw=.6))
for xx in [17.2,19.1,21]: ax.add_patch(Circle((xx,23.3),.35,fc=P['grey'],ec='none'))
ax.plot([17.2,21],[20,20],color=P['grey'],lw=.7)
arrow((9.8,21.2),(15.3,21.2))
text(6.1,15,'Sensor',ha='center',size=5.5);text(19.1,15,'Reference',ha='center',size=5.5)
text(2,8.8,'18 instruments',size=6)
text(2,4.8,'3 UK sites · QUANT',size=5.5)

# The delivery contract is the conceptual contribution, not a new estimator.
coords=[29.7,38.7,46.3]
ax.plot([coords[0],coords[-1]],[23,23],color=P['grey'],lw=.6)
for x in coords: ax.plot(x,23,'o',ms=2.4,mfc=P['blue'],mec='white',mew=.25)
for x,label in zip(coords,['u','u + d','u + d + 1']): text(x,26,label,ha='center',size=5.5)
text(29.7,19.5,'Request',ha='center',size=5.5)
text(39,19.5,'Release',ha='center',size=5.5)
text(46.3,19.5,'Output',ha='center',size=5.5)
arrow((38.7,16.8),(38.7,13.7))
text(28,8.3,'Delivery-aware\ncovariance design',size=6)

# Two labelled comparisons avoid implying that the two rows share one controller.
text(54,23.5,'Immediate · fixed risk',size=5.5)
text(54,18.5,f'{values[0]:.3f} → {values[1]:.3f}',size=6.8,weight='bold')
text(54,12.5,'72 h · delay-aware',size=5.5)
text(54,7.5,f'{values[2]:.3f} → {values[3]:.3f}',size=6.8,weight='bold')
text(54,3.3,'RMSE (ppb); vs periodic',size=5.5)
for y in [18.5,7.5]: ax.plot([53.25,53.25],[y-1.2,y+1.2],color=P['blue'],lw=1.2)

for ext in ['pdf','svg']: fig.savefig(OUT/f'gagraphic.{ext}',dpi=1200)
fig.savefig(OUT/'gagraphic_master.png',dpi=1200)
fig.savefig(OUT/'gagraphic.png',dpi=220)
# Set the submission metadata without resampling a single pixel.
with Image.open(OUT/'gagraphic.png') as im:
    assert im.size==(660,295),im.size
    im.save(OUT/'gagraphic.png',dpi=(300,300),optimize=True)
fig.savefig(ROOT/'temp/graphical_abstract_preview.png',dpi=300)
caption=('Graphical abstract. Reference acquisition designed for actual delivery improves sustained NO2 '
    'calibration in public QUANT records from 18 physical instruments across three UK sites. '
    'At the immediate shared B48 setting, fixed-risk AR reduces mean instrument RMSE from 4.582 to '
    '4.161 ppb relative to periodic AR. At 72 hour delivery, the development-selected delay-aware '
    'planner reduces RMSE from 5.060 to 4.664 ppb relative to periodic AR. The comparisons use '
    'different acquisition objectives; they do not estimate the effect of delay by comparing rows.')
(OUT/'gagraphic_caption.txt').write_text(caption+'\n',encoding='utf-8')
receipt={'source':'manuscript/article_nature.json','source_sha256':hashlib.sha256((ROOT/'manuscript/article_nature.json').read_bytes()).hexdigest(),
    'values':values,'master_width_mm':76.2,'submission_pixels':[660,295],'submission_dpi':300,
    'pixel_resampling':False,'minimum_text_pt_master':5.5,'minimum_text_pixels_submission':5.5*220/72,
    'files':[{ 'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(OUT.iterdir())]}
(ROOT/'evidence/graphical_abstract_validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print('Graphical abstract:',values,'660 x 295 px, 300 dpi; editable PDF and SVG.')
