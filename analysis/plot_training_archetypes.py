"""Draw report Figure 3 from the local log or the published CSV and metadata.

Run from the project root: python analysis/plot_training_archetypes.py
The full workspace uses logs/run_hydra_15038_r8m.log; the public repository
uses reports/training_archetypes.csv and .json, extracted from that log.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'logs/run_hydra_15038_r8m.log'
OUT = ROOT / 'reports/training_archetypes'
NAMES = {
    'dragapult': 'Dragapult', 'alakazam': 'Alakazam',
    'ogerpon_hydrapple': 'Ogerpon-Hydrapple', 'ogerpon': 'Ogerpon',
    'mega_lopunny': 'M-Lopunny', 'mega_lucario': 'M-Lucario',
    'marnie_grimmsnarl': "Marnie's Grimmsnarl",
    'cynthia': "Cynthia's Garchomp", 'ns_zoroark': "N's Zoroark",
    'kangaskhan_box': 'M-Kangaskhan box',
    'kangaskhan_crustle': 'M-Kangaskhan + Crustle',
    'grookey_dipplin': 'Grookey-Dipplin',
}
if LOG.exists():
    rows = [ast.literal_eval(line.strip()) for line in LOG.read_text(encoding='utf-8').splitlines()
            if line.strip().startswith("{'upd'")]
    source_hash = hashlib.sha256(LOG.read_bytes()).hexdigest()
    best_update = rows[-1]['best'][1]
    assert best_update == max(rows, key=lambda r: r['wr_pool'])['upd']
else:
    metadata = json.loads(OUT.with_suffix('.json').read_text(encoding='utf-8'))
    source_hash = metadata['source_sha256']
    best_update = metadata['best_update']
    inverse = {v: 'bc:' + k for k, v in NAMES.items()}
    with OUT.with_suffix('.csv').open(encoding='utf-8', newline='') as f:
        rows = [{'upd': int(r['update']),
                 'wr_arch': {inverse[k]: float(v) for k, v in r.items() if k != 'update'}}
                for r in csv.DictReader(f)]
assert [r['upd'] for r in rows] == list(range(1, 58))
keys = sorted({k for r in rows for k in r['wr_arch']
               if k.startswith('bc:') and not k.startswith('bc:mut_')})
assert all(k in r['wr_arch'] for r in rows for k in keys)
assert all(0 <= r['wr_arch'][k] <= 1 for r in rows for k in keys)
keys.sort(key=lambda k: rows[-1]['wr_arch'][k], reverse=True)
assert all(rows[-1]['wr_arch'][k] > rows[0]['wr_arch'][k] for k in keys)
assert all(r['wr_arch']['bc:alakazam'] == min(r['wr_arch'][k] for k in keys) for r in rows)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#c6c9cd', 'text.color': '#252a30',
                     'axes.labelcolor': '#555d66', 'xtick.color': '#555d66',
                     'ytick.color': '#555d66', 'svg.fonttype': 'none'})
ncols = 3
nrows = (len(keys) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 2.3 + 1.2),
                         sharex=True, sharey=True, squeeze=False)
for ax, key in zip(axes.flat, keys):
    values = [r['wr_arch'][key] * 100 for r in rows]
    ax.set_title(NAMES[key[3:]], loc='left', fontsize=11, fontweight='medium')
    ax.plot([r['upd'] for r in rows], values, color='#2a78d6', lw=1.65, zorder=3)
    ax.axvline(best_update, color='#a1a5ab', lw=1, ls=(0, (3, 3)), zorder=1)
    ax.scatter([57], [values[-1]], s=19, color='#2a78d6', zorder=4)
    ax.text(.97, .05, f'{values[0]:.1f}% to {values[-1]:.1f}%',
            transform=ax.transAxes, ha='right', fontsize=9, color='#555d66')
    ax.set_xlim(1, 58)
    ax.set_ylim(0, 100)
    ax.set_xticks([1, 15, 30, 45, 57])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis='y', color='#e6e8eb', lw=.7, zorder=0)
for ax in list(axes.flat)[len(keys):]:
    ax.set_visible(False)
fig.suptitle('Training win rate against each BC archetype', fontsize=18, x=.065,
             ha='left', y=.98)
fig.text(.065, .944, 'Final PPO run: updates 1-57. Each panel shows one fixed BC archetype.',
         fontsize=10, color='#555d66')
fig.supxlabel('PPO update', y=.065, fontsize=11)
fig.supylabel('Training win rate', x=.012, fontsize=11)
fig.text(.065, .025,
         'Each archetype pools the latest up to 300 games per exact decklist; draws count as half a win.\n'
         'Scripts, mutated lists and self-play excluded. Dashed line: peak overall training win rate (update 48).',
         fontsize=9, color='#555d66')
fig.subplots_adjust(left=.075, right=.975, top=.895, bottom=.12, hspace=.43, wspace=.17)
for suffix in ('png', 'svg', 'pdf'):
    fig.savefig(OUT.with_suffix('.' + suffix), dpi=190, facecolor='white')
plt.close(fig)
with OUT.with_suffix('.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['update'] + [NAMES[k[3:]] for k in keys])
    writer.writerows([[r['upd']] + [r['wr_arch'][k] for k in keys] for r in rows])
audit = {'source': str(LOG.relative_to(ROOT)),
         'source_sha256': source_hash, 'best_update': best_update,
         'updates': len(rows), 'archetypes': len(keys), 'missing_values': 0,
         'statistic': 'Logged wr_arch, pooling per-deck rolling windows; not arena-frequency weighted within archetype.',
         'smoothing': 'None added; source already uses per-deck rolling windows.',
         'start_end_percent': {NAMES[k[3:]]: [rows[0]['wr_arch'][k]*100, rows[-1]['wr_arch'][k]*100]
                               for k in keys}}
OUT.with_suffix('.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
print(json.dumps(audit, indent=2))
