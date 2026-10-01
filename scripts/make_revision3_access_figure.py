"""Simplify the access display using frozen original-budget results only."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'results/revision/diagnostics/error_total_access_curves.csv'
frame = pd.read_csv(source)
methods = [('periodic', 'AR periodic', '#333333', 'o'),
           ('risk_fixed', 'AR fixed risk', '#0072B2', 's'),
           ('risk', 'AR variable risk', '#56B4E9', '^'),
           ('gp_stream', 'GP periodic', '#009E73', 'D'),
           ('hgb_stream', 'Boosting periodic', '#D55E00', 'v')]
points = frame[(frame.lambda_hours == 0) & frame.method.isin([x[0] for x in methods])].copy()
assert len(points) == 15 and points.n_instruments.eq(18).all()
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(6.4, 2.9))
for method, label, color, marker in methods:
    part = points[points.method.eq(method)].sort_values('budget')
    assert list(part.budget) == [12, 24, 48]
    ax.plot(part.total_attempted_hours, part.macro_instrument_rmse,
            color=color, marker=marker, markersize=4, linewidth=1.4, label=label)
ax.set_xlabel('Total attempted reference hours, including commissioning')
ax.set_ylabel('Mean instrument RMSE (ppb)')
ax.set_xticks([2403, 2946, 4031], ['2,403\nB12', '2,946\nB24', '4,031\nB48'])
ax.grid(axis='y', alpha=.18)
ax.legend(loc='lower left', fontsize=8, ncol=2, frameon=False)
ax.set_ylim(4.07, 4.99)
fig.tight_layout()
out = ROOT/'results/figures'
fig.savefig(out/'revision3_figure1_access.svg')
fig.savefig(out/'revision3_figure1_access.png', dpi=900)
points[['budget','method','total_attempted_hours','macro_instrument_rmse']].sort_values(['method','budget']).to_csv(out/'revision3_figure1_data.csv', index=False)
print('Saved simplified Figure 1; 15 frozen curve points, no interpolation or new model selection.')
