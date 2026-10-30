"""Part 3: the game as an environment class and two Monte Carlo control agents (Numba jitclasses).

The environment steps through the exact transition table from solve.py, so one step is a lookup.
The agent sees only a state index and the reward (land gained this tick)."""
import numpy as np
from numba import int64, float64, njit
from numba.experimental import jitclass


@jitclass([('ptr', int64[:]), ('out', int64[:]), ('area', int64[:]), ('s', int64),
           ('t', int64), ('max_steps', int64)])
class TerritoryEnv:
    """Episodes end when the player dies (hunter on the trail, wall, own trail) or after max_steps."""

    def __init__(self, ptr, out, area, max_steps):
        self.ptr = ptr; self.out = out; self.area = area
        self.max_steps = max_steps; self.s = 0; self.t = 0

    def n_obs(self):
        return self.area.shape[0]

    def reset(self, s):
        self.s = s; self.t = 0
        return s

    def step(self, a):
        """Returns (next state or -1 if dead, reward, done)."""
        k = 4 * self.s + a
        lo, hi = self.ptr[k], self.ptr[k + 1]
        o = self.out[lo + np.random.randint(hi - lo)]          # several outcomes only on a respawn
        self.t += 1
        if o < 0: return -1, (-o - 1) - self.area[self.s], True
        r = self.area[o] - self.area[self.s]
        self.s = o
        return o, r, self.t >= self.max_steps


@jitclass([('Q', float64[:, :]), ('N', int64[:, :]), ('stamp', int64[:, :]), ('episode', int64),
           ('eps0', float64), ('eps1', float64), ('decay', float64), ('exploring', int64)])
class MCControl:
    """First-visit MC control with sample averages (gamma = 1).
    exploring = 0: on-policy epsilon-soft, every episode from the real start, epsilon decays
                   linearly from eps0 to eps1 over `decay` episodes.
    exploring = 1: exploring starts (Section 5.3): a uniformly random state and first move,
                   then greedy."""

    def __init__(self, n_obs, exploring, eps0, eps1, decay):
        self.Q = np.zeros((n_obs, 4)); self.N = np.zeros((n_obs, 4), np.int64)
        self.stamp = np.full((n_obs, 4), -1, np.int64)
        self.episode = 0; self.exploring = exploring
        self.eps0 = eps0; self.eps1 = eps1; self.decay = decay

    def greedy(self, s):
        best, cnt, pick = -np.inf, 0, 0
        for a in range(4):
            q = self.Q[s, a]
            if q > best: best, cnt, pick = q, 1, a
            elif q == best:
                cnt += 1
                if np.random.randint(cnt) == 0: pick = a        # uniform among ties
        return pick

    def start(self, n_obs):
        """Returns (start state, forced first move or -1)."""
        if self.exploring: return np.random.randint(n_obs), np.random.randint(4)
        return 0, -1

    def act(self, s):
        if not self.exploring:
            eps = self.eps0 + (self.eps1 - self.eps0) * min(1.0, self.episode / self.decay)
            if np.random.random() < eps: return np.random.randint(4)
        return self.greedy(s)

    def update(self, S, A, R, T):
        e = self.episode
        first = np.empty(T, np.bool_)
        for t in range(T):
            first[t] = self.stamp[S[t], A[t]] != e
            self.stamp[S[t], A[t]] = e
        G = 0.0
        for t in range(T - 1, -1, -1):
            G += R[t]
            if first[t]:
                s, a = S[t], A[t]
                self.N[s, a] += 1
                self.Q[s, a] += (G - self.Q[s, a]) / self.N[s, a]
        self.episode += 1


@njit(cache=False)
def greedy_game(env, agent, moves):
    """Final land of one greedy game from the real start; the moves are written to `moves`."""
    s = env.reset(0); land = env.area[0]; T = 0
    while True:
        a = agent.greedy(s); moves[T] = a; T += 1
        s2, r, done = env.step(a)
        land += r
        if done: return land, T
        s = s2


@njit(cache=False)
def train(env, agent, episodes, every):
    """Returns greedy final land and the fraction of states visited at each checkpoint."""
    cap = env.max_steps
    S = np.empty(cap, np.int64); A = np.empty(cap, np.int64); R = np.empty(cap, np.int64)
    moves = np.empty(cap, np.int64)
    m = episodes // every
    land = np.empty(m, np.int64); seen = np.empty(m)
    for e in range(episodes):
        s, a0 = agent.start(env.n_obs())
        s = env.reset(s); T = 0
        while True:
            a = a0 if T == 0 and a0 >= 0 else agent.act(s)
            s2, r, done = env.step(a)
            S[T] = s; A[T] = a; R[T] = r; T += 1
            if done: break
            s = s2
        agent.update(S, A, R, T)
        if (e + 1) % every == 0:
            i = (e + 1) // every - 1
            land[i] = greedy_game(env, agent, moves)[0]
            c = 0
            for j in range(env.n_obs()):
                if agent.N[j].sum() > 0: c += 1
            seen[i] = c / env.n_obs()
    return land, seen
