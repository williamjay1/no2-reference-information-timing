"""Reuse the retained vector layout with corrected scientific comparators.

The baseline generator is immutable; this adapter explicitly changes source,
output and captions, while preserving all vector primitives and print geometry.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'scripts/make_graphical_abstract.py').read_text(encoding='utf-8')
source=source.replace('results/graphical_abstract','results/graphical_abstract_revision6')
source=source.replace('article_nature.json','article_revision6.json')
loader="""import pandas as pd
old=pd.read_csv(ROOT/'results/revision/diagnostics/error_total_access_curves.csv')
old=old[(old.budget==48)&(old.lambda_hours==0)].set_index('method')
new=pd.read_csv(ROOT/'results/revision6/fixed_delay_corrected/matrix_aggregate_metrics.csv')
new=new[(new.shared==True)&(new.delay==72)].set_index('method')
data={'blocks':[{'type':'table','caption':'Table 2.','rows':[['AR','',round(old.loc['periodic','macro_instrument_rmse'],3),round(old.loc['risk_fixed','macro_instrument_rmse'],3)]]},
 {'type':'table','caption':'Table 3.','rows':[['72h','',round(new.loc['original','macro_instrument_rmse'],3),round(new.loc['aware_retained','macro_instrument_rmse'],3)]]}]}
"""
old_loader="data=json.loads((ROOT/'manuscript/article_revision6.json').read_text(encoding='utf-8'))"
assert source.count(old_loader)==1
source=source.replace(old_loader,loader)
source=source.replace("'source':'manuscript/article_revision6.json'","'source':'results/revision6/fixed_delay_corrected/matrix_aggregate_metrics.csv'")
source=source.replace("hashlib.sha256((ROOT/'manuscript/article_revision6.json').read_bytes()).hexdigest()",
    "hashlib.sha256((ROOT/'results/revision6/fixed_delay_corrected/matrix_aggregate_metrics.csv').read_bytes()).hexdigest()")
a=source.index('# Use the 72 hour');b=source.index('fig=plt.figure',a)
source=source[:a]+'''late=[r for r in table3['rows'] if r[0]=='72h']
assert len(late)==1
late=late[0]
values=[float(immediate[2]),float(immediate[3]),float(late[2]),float(late[3])]
assert values==[4.582,4.161,5.264,4.664]

'''+source[b:]
source=source.replace('Lower NO','Observed NO').replace('72 h · delay-aware','72 h · original / aware')
source=source.replace('RMSE (ppb); vs periodic','RMSE (ppb)')
a=source.index('caption=(');b=source.index("(OUT/'gagraphic_caption.txt')",a)
source=source[:a]+'''caption=('Graphical abstract. Reference-window planning explicitly accounts for event time and reference release. Public QUANT observations cover 18 physical instruments at three UK sites. Immediate shared correction gives 4.582 versus 4.161 ppb for periodic and original fixed-risk AR on full support (point estimates). At fixed 72-hour delivery, original-objective and retained delivery-aware AR give 5.264 and 4.664 ppb on identical common support; the paired contrast is supported by the stated retrospective seven-comparison Holm analysis. These rows have different supports and comparators and do not estimate a delay effect by subtraction. Randomized online release does not establish an aware-over-original advantage. No prospective field-cost benefit is claimed.')
'''+source[b:]
source=source.replace('temp/graphical_abstract_preview.png','temp/revision6_graphical_abstract_preview.png')
source=source.replace('evidence/graphical_abstract_validation.json','evidence/revision6/graphical_abstract_validation.json')
exec(compile(source,str(ROOT/'scripts/make_graphical_abstract.py'),'exec'),globals())
