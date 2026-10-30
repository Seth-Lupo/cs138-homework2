"""Runs every experiment and saves raw results under results/. Figures come from figures.py.

  python code/run.py            # all three parts
  python code/run.py --part 3   # one part
"""
import argparse
import json
import os
from collections import deque
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
SEED = 138
WORKERS = 8

P1 = dict(eps=[0.05, 0.1, 0.3], runs=10, episodes=200_000, every=1_000, noise=0.1, cap=5_000)
P2 = dict(radius=[0, 1, 2], encoding=[0, 1, 2], runs=5, episodes=10_000_000, every=500_000,
          eps=0.1, noise=0.1, other_noise=0.5, eval_games=500, cap=5_000)
P3 = dict(runs=10, episodes=20_000_000, every=200_000, max_steps=100, eps0=0.2, eps1=0.01, decay=0.8)


# ---------------------------------------------------------------- Part 1 and 2: racetrack
def _race_kernels():
    from numba import njit
    from racetrack import STARTS, episode, greedy_run

    @njit(cache=False)
    def train(env, agent, n, cap):
        S = np.empty(cap, np.int64); A = np.empty(cap, np.int64)
        R = np.empty(cap); P = np.empty(cap); M = np.empty((cap, 9), np.bool_)
        total = 0
        for _ in range(n): total += episode(env, agent, cap, S, A, R, P, M)
        return total / n

    @njit(cache=False)
    def every_start(env, agent, cap):
        tot = 0.0
        for col in STARTS: tot += greedy_run(env, agent, cap, col)
        return tot / len(STARTS)

    @njit(cache=False)
    def random_starts(env, agent, games, cap):
        steps = coll = 0.0
        for _ in range(games):
            steps += greedy_run(env, agent, cap, STARTS[np.random.randint(len(STARTS))])
            coll += env.collisions
        return steps / games, coll / games

    return train, every_start, random_starts


def trajectory(env, agent, col, cap=200, other=False):
    """Greedy positions from one start cell (env must expose the car(s) as attributes)."""
    from racetrack import legal
    s = env.reset_at(col)
    me = env.me if other else env.car
    path, opp = [(me.r, me.c)], [(env.other.r, env.other.c)] if other else []
    for _ in range(cap):
        mask = np.array([legal(me.vy, me.vx, a) for a in range(9)])
        s, r, done = env.step(agent.greedy(s, mask))
        if other: opp.append((env.other.r, env.other.c))
        if done: path.append(None); break
        path.append((me.r, me.c))
    return path, opp


def optimal_steps():
    """Fewest steps to the finish from each start cell with noise off (breadth-first search)."""
    from racetrack import ACC, ROWS, STARTS, RaceEnv, legal
    env = RaceEnv(0.0)
    best = []
    for col in STARTS:
        start = (ROWS - 1, int(col), 0, 0)
        dist, q = {start: 0}, deque([start])
        found = None
        while q and found is None:
            st = q.popleft()
            for a in range(9):
                if not legal(st[2], st[3], a): continue
                env.car.r, env.car.c, env.car.vy, env.car.vx = st
                _, _, done = env.step(a)
                if done: found = dist[st] + 1; break
                n = (env.car.r, env.car.c, env.car.vy, env.car.vx)
                if n[2] == 0 and n[3] == 0: continue          # crashed back to the start line
                if n not in dist: dist[n] = dist[st] + 1; q.append(n)
        best.append(found)
    return best


def part1_job(args):
    eps, run = args
    from racetrack import N_POS, OffPolicyMC, RaceEnv
    train, every_start, _ = _race_kernels()
    np.random.seed(SEED * 1000 + run)
    env, test, agent = RaceEnv(P1['noise']), RaceEnv(0.0), OffPolicyMC(N_POS, eps)
    k = P1['episodes'] // P1['every']
    greedy, behaviour = np.empty(k), np.empty(k)
    for i in range(k):
        behaviour[i] = train(env, agent, P1['every'], P1['cap'])
        greedy[i] = every_start(test, agent, 200)
    out = dict(greedy=greedy, behaviour=behaviour)
    if eps == 0.1 and run == 0:
        from racetrack import STARTS, make_policy
        out['policy'] = make_policy(agent)
        out['paths'] = [trajectory(test, agent, int(c))[0] for c in STARTS]
    return eps, run, out


def part1(pool):
    res = pool.map(part1_job, [(e, r) for e in P1['eps'] for r in range(P1['runs'])])
    greedy = np.array([[o['greedy'] for e2, r, o in res if e2 == e] for e in P1['eps']])
    behaviour = np.array([[o['behaviour'] for e2, r, o in res if e2 == e] for e in P1['eps']])
    best = next(o for e, r, o in res if e == 0.1 and r == 0)
    d = os.path.join(RES, 'part1'); os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, 'curves.npz'), greedy=greedy, behaviour=behaviour, policy=best['policy'],
             optimal=np.array(optimal_steps()))
    json.dump(best['paths'], open(os.path.join(d, 'paths.json'), 'w'))
    json.dump(P1, open(os.path.join(d, 'config.json'), 'w'), indent=1)


def part2_job(args):
    radius, enc, run = args
    from racetrack import OffPolicyMC, TwoCarEnv
    train, _, random_starts = _race_kernels()
    pol = np.load(os.path.join(RES, 'part1', 'curves.npz'))['policy']
    np.random.seed(SEED * 1000 + 100 * radius + 10 * enc + run)
    env = TwoCarEnv(pol, P2['noise'], P2['other_noise'], radius, enc)
    test = TwoCarEnv(pol, P2['noise'], P2['other_noise'], radius, enc)
    agent = OffPolicyMC(env.n_obs(), P2['eps'])
    k = P2['episodes'] // P2['every']
    steps, coll = np.empty(k), np.empty(k)
    for i in range(k):
        train(env, agent, P2['every'], P2['cap'])
        steps[i], coll[i] = random_starts(test, agent, P2['eval_games'], 300)
    out = dict(steps=steps, coll=coll)
    if radius == 2 and run == 0:
        np.random.seed(SEED)
        out['race'] = trajectory(test, agent, 5, other=True)
    return radius, enc, run, out


def part2(pool):
    jobs = [(r, e, k) for r in P2['radius'] for e in P2['encoding'] for k in range(P2['runs'])]
    res = pool.map(part2_job, sorted(jobs, key=lambda j: -j[1]))     # big tables first
    get = lambda key: np.array([[[o[key] for r2, e2, k, o in sorted(res) if r2 == r and e2 == e]
                                 for e in P2['encoding']] for r in P2['radius']])
    d = os.path.join(RES, 'part2'); os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, 'curves.npz'), steps=get('steps'), coll=get('coll'))
    json.dump({f'enc{e}': o['race'] for r, e, k, o in res if r == 2 and k == 0},
              open(os.path.join(d, 'races.json'), 'w'))
    json.dump(P2, open(os.path.join(d, 'config.json'), 'w'), indent=1)


# ---------------------------------------------------------------- Part 3: territory game
def part3_job(args):
    exploring, run = args
    from territory.mc import MCControl, TerritoryEnv, greedy_game, train
    z = np.load(os.path.join(RES, 'part3', 'table.npz'))
    np.random.seed(SEED * 1000 + 100 * exploring + run)
    env = TerritoryEnv(z['ptr'], z['out'], z['area'], P3['max_steps'])
    agent = MCControl(env.n_obs(), exploring, P3['eps0'], P3['eps1'], P3['decay'] * P3['episodes'])
    land, seen = train(env, agent, P3['episodes'], P3['every'])
    moves = np.empty(P3['max_steps'], np.int64)
    final, T = greedy_game(env, agent, moves)
    q_start = agent.Q[0].copy()
    return exploring, run, dict(land=land, seen=seen, moves=moves[:T].tolist(), final=int(final),
                                q_start=q_start)


def part3(pool):
    from territory.solve import build
    d = os.path.join(RES, 'part3')
    n, opt, line = build(d)
    print(f'part 3: {n} reachable states, optimum {opt} cells in {len(line)} ticks', flush=True)
    res = pool.map(part3_job, [(x, r) for x in (1, 0) for r in range(P3['runs'])])
    land = np.array([[o['land'] for x2, r, o in sorted(res) if x2 == x] for x in (0, 1)])
    seen = np.array([[o['seen'] for x2, r, o in sorted(res) if x2 == x] for x in (0, 1)])
    np.savez(os.path.join(d, 'curves.npz'), land=land, seen=seen)
    json.dump({f'{["eps", "es"][x]}{r}': dict(moves=o['moves'], final=o['final'])
               for x, r, o in res}, open(os.path.join(d, 'games.json'), 'w'))
    json.dump(dict(P3, states=n, optimum=int(opt), line=list(map(int, line))),
              open(os.path.join(d, 'config.json'), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, choices=[1, 2, 3])
    parts = [ap.parse_args().part] if ap.parse_args().part else [1, 2, 3]
    with Pool(WORKERS) as pool:
        for p in parts:
            {1: part1, 2: part2, 3: part3}[p](pool)
            print(f'part {p} done', flush=True)
