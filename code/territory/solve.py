"""Exact solution of the 5x5 game: enumerate every reachable state, then compute the most land the
player can guarantee from each one (worst case over hunter respawns), and one fastest optimal line.

Saves results/territory/table.npz, used by the MC environment:
  ptr, out   outcomes of (state s, move a) are out[ptr[4s+a] : ptr[4s+a+1]];
             >= 0 is a next state, < 0 is death with land -o-1
  area       land in each state
  value      guaranteed final land from each state (the exact optimum)"""
import os
import numpy as np
from .game import START, WAIT, actions, area, step

INF = 10 ** 9


def enumerate_states():
    idx, S, T = {START: 0}, [START], []
    i = 0
    while i < len(S):
        s = S[i]; i += 1; row = []
        for a in actions(s):
            outs = []
            for n in step(s, a):
                if n[0] == 'dead': outs.append(-bin(n[1]).count('1') - 1)
                else:
                    if n not in idx: idx[n] = len(S); S.append(n)
                    outs.append(idx[n])
            row.append((a, outs))
        T.append(row)
    return S, T


def solve_values(S, T):
    """Least fixed point of V(s) = max(area(s), max_a min_outcome V): the land the player can force."""
    V = [area(s) for s in S]
    val = lambda o: -o - 1 if o < 0 else V[o]
    changed = True
    while changed:
        changed = False
        for i in range(len(S) - 1, -1, -1):
            b = max(min(val(o) for o in outs) for _, outs in T[i])
            if b > V[i]: V[i] = b; changed = True
    return V


def fastest_line(S, T, k):
    """Attractor ranks for 'reach >= k land', then the move sequence that minimises worst-case time."""
    n = len(S); rank = [INF] * n
    pred = [[] for _ in range(n)]
    for i, row in enumerate(T):
        for ai, (_, outs) in enumerate(row):
            for o in set(outs):
                if o >= 0: pred[o].append((i, ai))
    left = [[len(set(o for o in outs if o >= 0)) for _, outs in row] for row in T]
    ok = [[all(-o - 1 >= k for o in outs if o < 0) for _, outs in row] for row in T]
    frontier = [i for i in range(n) if area(S[i]) >= k]
    for i in frontier: rank[i] = 0
    while frontier:
        nxt = []
        for j in frontier:
            for i, ai in pred[j]:
                left[i][ai] -= 1
                if rank[i] == INF and left[i][ai] == 0 and ok[i][ai]:
                    rank[i] = 1 + max(rank[o] for o in T[i][ai][1] if o >= 0); nxt.append(i)
        frontier = nxt
    line, i = [], 0
    while area(S[i]) < k:
        a, outs = min(((a, outs) for a, outs in T[i] if all(o >= 0 and rank[o] < INF for o in outs)),
                      key=lambda r: max(rank[o] for o in r[1]))
        line.append(a); i = outs[0]                    # no respawn happens on the optimal line
    return line


def build(out_dir):
    S, T = enumerate_states()
    V = solve_values(S, T)
    ptr, out = [0], []
    for s, row in zip(S, T):
        moves = dict(row)
        for a in range(4):                              # the agent always picks one of 4 directions
            out += moves[a] if a in moves else moves[s[3]] if s[3] != WAIT else moves[WAIT]
            ptr.append(len(out))
    line = fastest_line(S, T, V[0])
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, 'table.npz'), ptr=np.array(ptr), out=np.array(out),
                        area=np.array([area(s) for s in S]), value=np.array(V),
                        line=np.array(line), states=np.array([[s[0], s[2], s[3], s[4]] for s in S]),
                        trails=np.array([sum(1 << c for c in s[1]) for s in S]))
    return len(S), V[0], line
