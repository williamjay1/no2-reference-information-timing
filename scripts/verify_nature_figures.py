"""Verify scientific data and final-size geometry of the Nature-style figures.

Rebuilds graphics from frozen CSVs without fitting models or recomputing statistics.
Checks figure-level text, every axes title, ticks, legends and schematic arrows;
checks delivered vector/raster dimensions and fonts at their insertion widths.
Run: python scripts/verify_nature_figures.py
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'temp/matplotlib_nature'))
os.environ.setdefault('TEMP', str(ROOT / 'temp'))
os.environ.setdefault('TMP', str(ROOT / 'temp'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
from PIL import Image
import numpy as np
import pandas as pd
import pymupdf
import make_nature_figures as mf

FROZEN = ROOT / 'delivery/NO2_Reference_Timing_Delivery'
SOURCE_MAP = {
    'results/revision4/components/weekly_coefficients_uncertainty.csv':
        'results/components/weekly_coefficients_uncertainty.csv',
    'results/revision/diagnostics/B48_fixed_vs_periodic_instruments.csv':
        'results/revision/B48_fixed_vs_periodic_instruments.csv',
    'results/revision/branch/all_points.csv': 'results/revision/branch_all_points.csv',
    'results/revision/branch/summary.csv': 'results/revision/branch_summary.csv',
    'results/revision/diagnostics/error_total_access_curves.csv':
        'results/revision/error_total_access_curves.csv',
    'results/revision/diagnostics/minimum_tested_budget_frontier.csv':
        'results/revision/minimum_tested_budget_frontier.csv',
}
# Recorded from the preserved revision-5 delivery before presentation changes.
# Keeping these immutable checksums in the verifier also permits an extracted
# new delivery to verify its numerical sources without the older package folder.
FROZEN_HASHES = dict(zip(SOURCE_MAP, [
    'affa8fbf61bb6e825b08532f28321978d1c416b6342b22b67da2070bd49a2dfb',
    '752307581c2905897679f9cb30aa39c8026736ef8e0211a48a797895b1a32911',
    'a38981fc1088b9699449f2b85c7ddda3554575c8dac3d2f15853ab9babd23103',
    '7bb6d06ad0e0674c76136f3ccf405db9dc472ca93c589eeca007f20eb2d94c8c',
    'fa7082847fe6be02b898559e07fdf1257fce344c2f44ed4b0df353951f6561d4',
    'f93a1e4c706a6a5c059ca65df3f4e698901baaee888bbf3415d9dc7d98c013f9',
]))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def texts(fig):
    """Collect visible Text once, including figure text and left/right titles."""
    seen = set()
    entries = []

    def add(artist, category):
        if artist is None or id(artist) in seen or not artist.get_visible():
            return
        seen.add(id(artist))
        if not artist.get_text().strip():
            return
        entries.append((artist, category))

    for text in fig.texts:
        add(text, 'figure text')
    for ai, ax in enumerate(fig.axes):
        for text in ax.texts:
            add(text, f'axes {ai} text')
        for title in (ax.title, getattr(ax, '_left_title', None),
                      getattr(ax, '_right_title', None)):
            add(title, f'axes {ai} title')
        if ax.axison:
            for text in (ax.xaxis.label, ax.yaxis.label,
                         ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()):
                add(text, f'axes {ai} label')
            for axis in (ax.xaxis, ax.yaxis):
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    add(tick.label1, f'axes {ai} tick')
                    add(tick.label2, f'axes {ai} tick')
        lg = ax.get_legend()
        if lg is not None:
            add(lg.get_title(), f'axes {ai} legend title')
            for text in lg.get_texts():
                add(text, f'axes {ai} legend')
    for li, lg in enumerate(fig.legends):
        add(lg.get_title(), f'figure legend {li} title')
        for text in lg.get_texts():
            add(text, f'figure legend {li}')
    return entries


def short(artist):
    return artist.get_text().replace('\n', ' / ')[:100]


def check_geometry(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    boxes = [(artist, category, artist.get_window_extent(renderer))
             for artist, category in texts(fig)]
    overlaps = []
    for i, (a, ca, ba) in enumerate(boxes):
        for b, cb, bb in boxes[i + 1:]:
            dx = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            dy = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if dx > .5 and dy > .5:
                overlaps.append({'first': short(a), 'second': short(b),
                                 'categories': [ca, cb],
                                 'overlap_px': [round(dx, 2), round(dy, 2)]})
    w, h = canvas.get_width_height()
    out_of_canvas = []
    for text, category, bb in boxes:
        if bb.x0 < -.2 or bb.y0 < -.2 or bb.x1 > w + .2 or bb.y1 > h + .2:
            out_of_canvas.append({'text': short(text), 'category': category,
                                  'bbox_px': [round(v, 2) for v in bb.extents]})
    arrow_hits = []
    arrow_count = 0
    for ai, ax in enumerate(fig.axes):
        if ax.axison:
            continue
        # Schematic card boundaries intentionally contain their text. Only actual
        # arrows are checked for crossing a text box; rectangles are not errors.
        for patch in ax.patches:
            if not isinstance(patch, FancyArrowPatch) or not patch.get_visible():
                continue
            arrow_count += 1
            path = patch.get_path().transformed(patch.get_transform())
            # FancyArrowPatch terminates its closed arrowhead with a dummy (0,0)
            # CLOSEPOLY vertex. intersects_bbox on that compound path can falsely
            # connect the head to the canvas origin. Flatten valid subpaths first.
            segments = [MplPath(poly) for poly in path.to_polygons(closed_only=False)]
            for text, category, bb in boxes:
                inner = bb.padded(-.5)
                if any(segment.intersects_bbox(inner, filled=False) for segment in segments):
                    arrow_hits.append({'axes': ai, 'arrow': arrow_count,
                                       'text': short(text), 'category': category})
    return {'text_artists': len(boxes), 'arrow_count': arrow_count,
            'text_overlaps': overlaps, 'text_outside_canvas': out_of_canvas,
            'arrows_crossing_text': arrow_hits,
            'text_sizes_pt': sorted(set(float(a.get_fontsize()) for a, _, _ in boxes))}


def equal_xy(actual, expected):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    return actual.shape == expected.shape and np.allclose(actual, expected,
        rtol=0, atol=1e-12, equal_nan=True)


def one_point_lines(ax):
    return [line for line in ax.lines if len(line.get_xdata()) == 1 and
            line.get_marker() not in ('None', '', ' ', None)]


def check_data(fig, stem):
    checks = []
    point_counts = {}

    def check(name, ok):
        checks.append({'check': name, 'pass': bool(ok)})

    if 'coefficient_trajectories' in stem:
        d = pd.read_csv(ROOT / next(iter(SOURCE_MAP)))
        d = d[d.family.eq('Zep') & ~d.deployment.eq('Manchester__Zep344__s2')]
        segments = ['Manchester__Zep188__s1', 'Manchester__Zep344__s1',
                    'London__Zep311__s2', 'York__Zep309__s1']
        fields = ['baseline_ppb', 'pollutant_gain', 'temperature_ppb_per_C',
                  'humidity_ppb_per_percent']
        total = 0
        for ax, field in zip(fig.axes, fields):
            check(field + ' has four trajectories', len(ax.lines) == 4)
            for line, seg in zip(ax.lines, segments):
                g = d[d.deployment.eq(seg) & d.coefficient.eq(field)].sort_values('elapsed_days')
                check(field + ' / ' + seg, equal_xy(line.get_xydata(),
                    g[['elapsed_days', 'mean']].to_numpy()))
                total += len(line.get_xdata())
        point_counts = {'plotted': total, 'source_filtered': len(d)}
        check('all weekly means represented', total == len(d))
    elif 'instrument_contrasts' in stem:
        d = pd.read_csv(ROOT / 'results/revision/diagnostics/B48_fixed_vs_periodic_instruments.csv')
        d['long'] = d.evidence_group.eq('long_Zep')
        d = d.sort_values(['long', 'site', 'instrument'],
                         ascending=[False, True, True]).reset_index(drop=True)
        points = one_point_lines(fig.axes[0])
        actual = np.array([line.get_xydata()[0] for line in points])
        expected = np.column_stack([d.rmse_difference, np.arange(len(d))])
        check('all instrument contrasts equal frozen values', equal_xy(actual, expected))
        point_counts = {'plotted': len(points), 'source': len(d)}
    elif 'local_window_value' in stem:
        d = pd.read_csv(ROOT / 'results/revision/branch/all_points.csv')
        s = pd.read_csv(ROOT / 'results/revision/branch/summary.csv')
        names = ['Zep188', 'Zep309', 'Zep311', 'Zep344']
        collections = fig.axes[0].collections
        check('four instrument point collections', len(collections) == 4)
        total = 0
        for coll, name in zip(collections, names):
            expected = d[d.instrument.eq(name)][
                ['predicted_difference', 'realized_mse_difference']].to_numpy()
            check(name + ' point values match', equal_xy(coll.get_offsets(), expected))
            total += len(coll.get_offsets())
        correlations = one_point_lines(fig.axes[1])
        expected = np.column_stack([s.set_index('instrument').loc[names, 'spearman_mse'],
                                   np.arange(4)])
        actual = np.array([line.get_xydata()[0] for line in correlations])
        check('four frozen correlation values', equal_xy(actual, expected))
        point_counts = {'scatter_plotted': total, 'scatter_source': len(d),
                        'correlations': len(correlations)}
        check('all eligible origins represented', total == len(d))
    elif 'access_tradeoff' in stem:
        c = pd.read_csv(ROOT / 'results/revision/diagnostics/error_total_access_curves.csv')
        f = pd.read_csv(ROOT / 'results/revision/diagnostics/minimum_tested_budget_frontier.csv')
        methods = ['periodic', 'risk_fixed', 'risk', 'gp_stream', 'hgb_stream', 'uncertainty']
        total = 0
        for ai, ax in enumerate(fig.axes):
            check(f'panel {ai} has six methods', len(ax.lines) == 6)
            for line, method in zip(ax.lines, methods):
                if ai == 0:
                    g = c[c.method.eq(method) & c.lambda_hours.eq(0)].sort_values('budget')
                    expected = g[['total_attempted_hours', 'macro_instrument_rmse']].to_numpy()
                else:
                    lam = 0 if ai == 1 else 72
                    g = f[f.method.eq(method) & f.lambda_hours.eq(lam) &
                          f.endpoint.eq('macro_instrument_rmse')].sort_values('target_ppb')
                    expected = g[['target_ppb', 'equivalent_hours']].to_numpy()
                check(f'panel {ai} / {method} data', equal_xy(line.get_xydata(), expected))
                total += len(line.get_xydata())
        point_counts = {'curve_and_frontier_vertices': total}
    else:
        check('schematic contains no estimated numerical result',
              all(not ax.axison for ax in fig.axes))
    return {'checks': checks, 'point_counts': point_counts,
            'status': 'PASS' if all(x['pass'] for x in checks) else 'FAIL'}


def check_delivered(stem, variant, width, height, raster=True):
    suffix = '_single' if variant == 'latex' and stem in mf.SCHEMATICS else ''
    folder = ROOT / f'results/figures_nature_{variant}'
    pdf_path = folder / f'{stem}{suffix}.pdf'
    png_path = folder / f'{stem}{suffix}.png'
    issues = []
    record = {'pdf': pdf_path.relative_to(ROOT).as_posix(),
              'png': png_path.relative_to(ROOT).as_posix()}
    if not pdf_path.exists():
        issues.append('vector PDF missing')
    else:
        with pymupdf.open(pdf_path) as pdf:
            size = [pdf[0].rect.width * 25.4 / 72, pdf[0].rect.height * 25.4 / 72]
            record['pdf_size_mm'] = size
            if pdf.page_count != 1 or abs(size[0] - width) > .02 or abs(size[1] - height) > .02:
                issues.append('PDF physical dimensions differ from insertion dimensions')
            fonts = [list(font) for page in pdf for font in page.get_fonts(full=True)]
            record['font_names'] = sorted(set(font[3] for font in fonts))
            if any(font[2].lower() == 'type3' for font in fonts):
                issues.append('Type 3 font present in vector PDF')
    if not png_path.exists():
        issues.append('raster PNG missing')
    elif raster:
        with Image.open(png_path) as im:
            pixel_width, pixel_height = im.size
            effective = pixel_width / (width / 25.4)
            record['png_pixels'] = [pixel_width, pixel_height]
            record['png_dpi_tag'] = list(im.info.get('dpi', (0, 0)))
            record['effective_dpi'] = effective
            if abs(effective - 1200) > 1.0:
                issues.append('PNG does not have 1200 effective dpi at insertion width')
            if abs(pixel_height / (height / 25.4) - 1200) > 1.0:
                issues.append('PNG height does not match physical figure height')
            if abs(record['png_dpi_tag'][0] - 1200) > 1.0:
                issues.append('PNG resolution metadata differs from 1200 dpi')
            ink = np.asarray(im.convert('L')) < 250
            edges = [name for name, arr in [('top', ink[0, :]), ('bottom', ink[-1, :]),
                                           ('left', ink[:, 0]), ('right', ink[:, -1])]
                     if arr.any()]
            record['ink_at_canvas_edges'] = edges
            if edges:
                issues.append('rendered ink touches canvas edge: ' + ', '.join(edges))
    record['issues'] = issues
    record['status'] = 'PASS' if not issues else 'FAIL'
    return record


def check_graphical_abstract(raster=True):
    script = ROOT / 'scripts/make_graphical_abstract.py'
    if not script.exists():
        return {'status': 'NOT_REQUESTED'}
    source = script.read_text(encoding='utf-8')
    # Execute only the figure construction, before any saving, image metadata
    # writes or receipt writes. This captures the actual submission design while
    # leaving every delivered graphical-abstract file untouched.
    prefix = source.split("for ext in ['pdf','svg']:", 1)[0]
    if prefix == source:
        raise ValueError('Graphical abstract construction/export boundary changed.')
    namespace = {'__file__': str(script), '__name__': '__geometry_capture__'}
    exec(compile(prefix, str(script), 'exec'), namespace)
    fig = namespace['fig']
    geometry = check_geometry(fig)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    separators = []
    for ax in fig.axes:
        for line in ax.lines:
            xy = line.get_xydata()
            if len(xy) != 2 or abs(xy[0, 0] - xy[1, 0]) > 1e-9 or \
                    abs(xy[0, 1] - xy[1, 1]) < 20:
                continue
            path = MplPath(line.get_transform().transform(xy))
            for text, category in texts(fig):
                if path.intersects_bbox(text.get_window_extent(renderer).padded(-.5), filled=False):
                    separators.append({'text': short(text), 'category': category})
    geometry['separator_crossings'] = separators
    min_size = min(a.get_fontsize() * fig.dpi / 72 for a, _ in texts(fig))
    issues = []
    values = namespace['values']
    if values != [4.582, 4.161, 5.060, 4.664]:
        issues.append('graphical abstract numbers changed')
    if min_size < 16:
        issues.append('submission text smaller than 16 pixels')
    folder = ROOT / 'results/graphical_abstract'
    pdf_file = folder / 'gagraphic.pdf'
    with pymupdf.open(pdf_file) as doc:
        widths = [p.rect.width * 25.4 / 72 for p in doc]
        if len(widths) != 1 or abs(widths[0] - 76.2) > .02:
            issues.append('graphical abstract PDF master is not 76.2 mm wide')
        fonts = [font for page in doc for font in page.get_fonts(full=True)]
        if any(font[2].lower() == 'type3' for font in fonts):
            issues.append('Type 3 font in graphical abstract master')
    with Image.open(folder / 'gagraphic.png') as im:
        pixels = list(im.size)
        dpi = list(im.info.get('dpi', [0, 0]))
        if pixels != [660, 295] or abs(dpi[0] - 300) > 1:
            issues.append('graphical abstract submission size or metadata changed')
        if raster:
            ink = np.asarray(im.convert('L')) < 250
            if ink[0, :].any() or ink[-1, :].any() or ink[:, 0].any() or ink[:, -1].any():
                issues.append('graphical abstract ink touches submission canvas edge')
    problems = geometry['text_overlaps'] or geometry['text_outside_canvas'] or \
        geometry['arrows_crossing_text'] or separators or issues
    record = {'geometry': geometry, 'master_width_mm': widths[0],
              'submission_pixels': pixels, 'submission_dpi_tag': dpi,
              'minimum_text_pixels': min_size, 'values': values,
              'font_names': sorted(set(font[3] for font in fonts)), 'issues': issues,
              'status': 'FAIL' if problems else 'PASS'}
    print(f"{record['status']:4s} graphical abstract: overlaps "
          f"{len(geometry['text_overlaps'])}, separator crossings {len(separators)}, "
          f"minimum text {min_size:.2f} px")
    for overlap in geometry['text_overlaps']:
        print('  overlap:', overlap['first'], '<>', overlap['second'])
    for crossing in separators:
        print('  separator crossing:', crossing['text'])
    plt.close(fig)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', choices=['word', 'latex'],
                        default=['word', 'latex'])
    parser.add_argument('--no-raster', action='store_true', help='quick live layout check')
    args = parser.parse_args()
    source_checks = []
    for source, archive in SOURCE_MAP.items():
        current = ROOT / source
        frozen = FROZEN / archive
        current_hash = digest(current)
        recorded_hash = FROZEN_HASHES[source]
        frozen_hash = digest(frozen) if frozen.exists() else recorded_hash
        receipt_hashes = []
        for variant in args.variants:
            receipt = json.loads((ROOT / f'evidence/nature_figures_{variant}.json').read_text())
            receipt_hashes.extend(s['sha256'] for s in receipt['sources'] if s['path'] == source)
        ok = recorded_hash == frozen_hash == current_hash and \
             len(receipt_hashes) == len(args.variants) and \
             all(h == current_hash for h in receipt_hashes)
        source_checks.append({'source': source, 'sha256': current_hash,
            'frozen_delivery_sha256': frozen_hash, 'recorded_revision5_sha256': recorded_hash,
            'historical_file_available': frozen.exists(),
            'generation_receipt_sha256': receipt_hashes,
            'pass': ok})
    report = {'scope': 'Read-only figure rebuild; no model training or statistical rerun.',
              'source_checks': source_checks, 'figures': [],
              'raster_checked': not args.no_raster}
    for variant in args.variants:
        for stem, builder in mf.BUILDERS:
            width = 175.26 if variant == 'word' else (88 if stem in mf.SCHEMATICS else 181)
            fig = builder(width)
            height = fig.get_size_inches()[1] * 25.4
            geometry = check_geometry(fig)
            data = check_data(fig, stem)
            delivered = check_delivered(stem, variant, width, height, not args.no_raster)
            problems = (geometry['text_overlaps'] or geometry['text_outside_canvas'] or
                        geometry['arrows_crossing_text'] or data['status'] != 'PASS' or
                        delivered['status'] != 'PASS')
            item = {'variant': variant, 'stem': stem, 'width_mm': width, 'height_mm': height,
                    'geometry': geometry, 'data': data, 'delivered': delivered,
                    'status': 'FAIL' if problems else 'PASS'}
            report['figures'].append(item)
            print(f"{item['status']:4s} {variant:5s} {stem}: "
                  f"text overlaps {len(geometry['text_overlaps'])}, "
                  f"outside {len(geometry['text_outside_canvas'])}, "
                  f"arrow hits {len(geometry['arrows_crossing_text'])}; data {data['status']}")
            for overlap in geometry['text_overlaps']:
                print('  overlap:', overlap['first'], '<>', overlap['second'])
            for issue in geometry['text_outside_canvas']:
                print('  outside:', issue['text'])
            for issue in geometry['arrows_crossing_text']:
                print('  arrow crossing:', issue['text'])
            for issue in delivered['issues']:
                print('  delivered:', issue)
            plt.close(fig)
    report['graphical_abstract'] = check_graphical_abstract(not args.no_raster)
    report['status'] = 'PASS' if all(s['pass'] for s in source_checks) and \
        all(f['status'] == 'PASS' for f in report['figures']) and \
        report['graphical_abstract']['status'] in ('PASS', 'NOT_REQUESTED') else 'FAIL'
    out = ROOT / 'evidence/nature_geometry.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"Frozen source checks: {sum(s['pass'] for s in source_checks)}/{len(source_checks)}")
    print('Status:', report['status'], '|', out)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
