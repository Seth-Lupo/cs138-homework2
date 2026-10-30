"""Builds every figure in results/figures/ from the saved results.  python code/figures.py"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from racetrack import GRID
from territory.game import START, WAIT, area, step

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(RES, 'figures')
C = ['#2a78d6', '#eb6834', '#1baf7a']                 # categorical slots 1-3, fixed order
INK, MUTED, GRIDC = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRIDC,
                     'grid.linewidth': 0.6, 'legend.frameon': False, 'lines.linewidth': 1.6})


def band(ax, x, runs, color, label, median=False):
    """Mean with a band of one standard error, or (median=True) median with the middle half of runs."""
    if median:
        m = np.median(runs, 0); lo, hi = np.percentile(runs, [25, 75], 0)
    else:
        m = runs.mean(0); se = runs.std(0, ddof=1) / np.sqrt(len(runs)); lo, hi = m - se, m + se
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(x, m, color=color, label=label)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), bbox_inches='tight'); plt.close(fig)


def track(ax):
    cmap = ListedColormap(['#3b3a37', '#f4f3ef', '#cfe3cf', '#f3d9c8'])
    ax.imshow(GRID, cmap=cmap, vmin=0, vmax=3, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def path_xy(path):
    """Columns and rows of a drive; a gap (NaN) where the car was sent back to the start line."""
    xs, ys, prev = [], [], None
    for p in path:
        if p is None: continue
        if prev is not None and p[0] > prev[0]: xs.append(np.nan); ys.append(np.nan)
        xs.append(p[1]); ys.append(p[0]); prev = p
    return xs, ys


# ---------------------------------------------------------------- Part 1
def part1():
    z = np.load(os.path.join(RES, 'part1', 'curves.npz')); cfg = json.load(open(os.path.join(RES, 'part1', 'config.json')))
    x = np.arange(1, z['greedy'].shape[2] + 1) * cfg['every'] / 1e3
    fig, ax = plt.subplots(figsize=(6.5, 3))
    for i, eps in enumerate(cfg['eps']):
        band(ax, x, z['greedy'][i], C[i], f'behaviour $\\epsilon={eps}$', median=True)
    ax.axhline(z['optimal'].mean(), color=MUTED, ls='--', lw=1)
    ax.text(x[-1], z['optimal'].mean() * 0.86, f"optimum {z['optimal'].mean():.1f}", ha='right', va='top', color=MUTED)
    ax.set_yscale('log'); ax.set_ylim(8, 210)
    ax.set_xlabel('training episodes (thousands)'); ax.set_ylabel('greedy steps to finish\n(median of 10 runs, 200 = never)')
    ax.legend(loc='upper right')
    save(fig, 'part1_curves.pdf')

    paths = json.load(open(os.path.join(RES, 'part1', 'paths.json')))
    fig, ax = plt.subplots(figsize=(3.2, 5.6))
    track(ax)
    for p in paths:
        xs, ys = path_xy(p)
        ax.plot(xs, ys, '-o', color=C[0], ms=2.5, lw=1.2)
        ax.annotate('', xy=(xs[-1] + 1, ys[-1]), xytext=(xs[-1], ys[-1]),
                    arrowprops=dict(arrowstyle='->', color=C[0], lw=1.2))
    save(fig, 'part1_track.pdf')


# ---------------------------------------------------------------- Part 2
def part2():
    z = np.load(os.path.join(RES, 'part2', 'curves.npz')); cfg = json.load(open(os.path.join(RES, 'part2', 'config.json')))
    names = ['opponent unseen', '+ relative position', '+ position and velocity sign']
    x = np.arange(1, z['steps'].shape[3] + 1) * cfg['every'] / 1e6
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.2), sharex=True, sharey='row')
    for j, r in enumerate(cfg['radius']):
        for e in cfg['encoding']:
            band(axes[0, j], x, z['steps'][j, e], C[e], names[e])
            band(axes[1, j], x, z['coll'][j, e], C[e], names[e])
        axes[0, j].set_title(f'collision zone {2 * r + 1}$\\times${2 * r + 1}', color=INK)
        axes[1, j].set_xlabel('episodes (millions)')
    axes[0, 0].set_ylabel('greedy steps to finish'); axes[1, 0].set_ylabel('collisions per race')
    axes[0, 0].legend(loc='upper right', fontsize=7)
    save(fig, 'part2_curves.pdf')

    races = json.load(open(os.path.join(RES, 'part2', 'races.json')))
    fig, axes = plt.subplots(1, 2, figsize=(5, 5))
    for ax, key, title in zip(axes, ['enc0', 'enc2'], ['opponent unseen', 'position and velocity sign']):
        track(ax); me, opp = races[key]
        xs, ys = path_xy(me); ax.plot(xs, ys, '-o', color=C[0], ms=2.5, lw=1.2, label='learner')
        xs, ys = path_xy(opp); ax.plot(xs, ys, '-s', color=C[1], ms=2.5, lw=1.2, label='opponent')
        ax.set_title(title, fontsize=8, color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=8)
    save(fig, 'part2_race.pdf')


# ---------------------------------------------------------------- Part 3
def replay(moves):
    """Boards after each tick of a move list (the solver's game rules)."""
    s, boards = START, []
    for a in moves:
        n = step(s, a)[0]
        if n[0] == 'dead': boards.append(('dead', s)); break
        s = n; boards.append(('ok', s))
    return boards


def board(ax, s, title, dead=False):
    own, path, me, d, f = s
    img = np.zeros((5, 5, 3))
    for c in range(25):
        col = '#e9e8e4'
        if own >> c & 1: col = '#9cc8e8'
        if c in path: col = '#2f7fc1'
        if c == me: col = '#0b4f8a'
        if c == f: col = '#d33a4a'
        img[c // 5, c % 5] = matplotlib.colors.to_rgb(col)
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    for k in np.arange(-0.5, 5, 1):
        ax.axhline(k, color='white', lw=2); ax.axvline(k, color='white', lw=2)
    ax.set_title(title, fontsize=8, color=INK)


def part3():
    z = np.load(os.path.join(RES, 'part3', 'curves.npz')); cfg = json.load(open(os.path.join(RES, 'part3', 'config.json')))
    x = np.arange(1, z['land'].shape[2] + 1) * cfg['every'] / 1e6
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.2), sharex=True)
    for i, name in enumerate(['on-policy $\\epsilon$-soft', 'exploring starts']):
        band(axes[0], x, z['land'][i].astype(float), C[i], name)
        band(axes[1], x, 100 * z['seen'][i], C[i], name)
    axes[0].axhline(cfg['optimum'], color=MUTED, ls='--', lw=1)
    axes[0].text(x[-1], cfg['optimum'] + 0.3, f"proven optimum {cfg['optimum']}", ha='right', va='bottom', color=MUTED)
    axes[0].set_ylim(0, 23); axes[0].set_ylabel('greedy final land\n(cells of 25)')
    axes[1].set_ylabel('states visited (%)'); axes[1].set_xlabel('training episodes (millions)')
    axes[0].legend(loc='lower right')
    save(fig, 'part3_curves.pdf')

    games = json.load(open(os.path.join(RES, 'part3', 'games.json')))
    opt = replay(cfg['line'])
    caps = [i for i in range(len(opt)) if i == 0 or area(opt[i][1]) != area(opt[i - 1][1])]
    best_es = max((v['final'], k) for k, v in games.items() if k.startswith('es'))[1]
    best_eps = max((v['final'], k) for k, v in games.items() if k.startswith('eps'))[1]
    fig, axes = plt.subplots(1, 8, figsize=(8, 1.5))
    for ax, i in zip(axes[:6], caps[-6:]):
        board(ax, opt[i][1], f'tick {i + 1}: {area(opt[i][1])}')
    for ax, (key, label) in zip(axes[6:], [(best_es, 'ES'), (best_eps, '$\\epsilon$-soft')]):
        last = replay(games[key]['moves'])[-1][1]
        board(ax, last, f'{label}: {area(last)}')
    save(fig, 'part3_games.pdf')


if __name__ == '__main__':
    for name, f in [('part1', part1), ('part2', part2), ('part3', part3)]:
        if os.path.exists(os.path.join(RES, name, 'curves.npz')): f(); print(name, 'figures done')
