"""Parts 1 and 2: the racetrack (Exercise 5.12), a two-car version, and off-policy MC control.

Environments and the agent are separate jitclasses. The agent sees only an observation index,
the reward, and which of the 9 actions are legal; it never reads the track or the other car."""
import numpy as np
from numba import int64, float32, float64, int8, boolean, njit
from numba.experimental import jitclass

# Left track of Figure 5.5. '#' off track, '.' track, 'S' start line, 'F' finish line.
TRACK = """\
###............F
##.............F
##.............F
#..............F
...............F
...............F
..........######
..........######
..........######
..........######
..........######
..........######
..........######
..........######
#.........######
#.........######
#.........######
#.........######
#.........######
#.........######
#.........######
#.........######
##........######
##........######
##........######
##........######
##........######
##........######
##........######
###.......######
###.......######
###SSSSSS.######"""

GRID = np.array([[{'#': 0, '.': 1, 'S': 2, 'F': 3}[ch] for ch in row] for row in TRACK.split('\n')], np.int8)
ROWS, COLS = GRID.shape
STARTS = np.array([c for c in range(COLS) if GRID[ROWS - 1, c] == 2], np.int64)
VMAX = 4
N_POS = ROWS * COLS * (VMAX + 1) ** 2               # (row, col, vy, vx)
ACC = np.array([(ay, ax) for ay in (-1, 0, 1) for ax in (-1, 0, 1)], np.int64)


@njit(cache=True)
def own_index(r, c, vy, vx):
    return ((r * COLS + c) * (VMAX + 1) + vy) * (VMAX + 1) + vx


@njit(cache=True)
def legal(vy, vx, a):
    """Velocity stays in 0..4 on both axes and is never (0, 0) after an action."""
    ny, nx = vy + ACC[a, 0], vx + ACC[a, 1]
    return 0 <= ny <= VMAX and 0 <= nx <= VMAX and (ny > 0 or nx > 0)


@njit(cache=True)
def point(r, c, vy, vx, k, n):
    """k-th of n evenly spaced points on the move from (r, c); up is -row, right is +col."""
    return r - int(round(vy * k / n)), c + int(round(vx * k / n))


@njit(cache=True)
def cell(r, c):
    """0 off track, 1 track, 3 finish. Passing the right edge in a finish row counts as finishing."""
    if r < 0 or r >= ROWS: return 0
    if c >= COLS: return 3 if GRID[r, COLS - 1] == 3 else 0
    if c < 0: return 0
    g = GRID[r, c]
    return 1 if g == 2 else g


car_spec = [('r', int64), ('c', int64), ('vy', int64), ('vx', int64)]


@jitclass(car_spec)
class Car:
    def __init__(self):
        self.r = 0; self.c = 0; self.vy = 0; self.vx = 0

    def to_start(self, col):
        self.r = ROWS - 1; self.c = col; self.vy = 0; self.vx = 0

    def index(self):
        return own_index(self.r, self.c, self.vy, self.vx)

    def accelerate(self, a, noise):
        if np.random.random() >= noise:                # with probability `noise` both increments are 0
            self.vy += ACC[a, 0]; self.vx += ACC[a, 1]


@jitclass([('car', Car.class_type.instance_type), ('noise', float64), ('steps', int64)])
class RaceEnv:
    """One car. Reward -1 per step; leaving the track sends the car to a random start cell."""

    def __init__(self, noise):
        self.car = Car(); self.noise = noise; self.steps = 0

    def n_obs(self):
        return N_POS

    def reset(self):
        return self.reset_at(STARTS[np.random.randint(len(STARTS))])

    def reset_at(self, col):
        self.car.to_start(col); self.steps = 0
        return self.car.index()

    def legal(self, a):
        return legal(self.car.vy, self.car.vx, a)

    def step(self, a):
        """Returns (observation, reward, finished)."""
        car = self.car; self.steps += 1
        car.accelerate(a, self.noise)
        n = max(car.vy, car.vx)
        if n == 0: return car.index(), -1.0, False      # noise kept a starting car at rest
        for k in range(1, n + 1):
            pr, pc = point(car.r, car.c, car.vy, car.vx, k, n)
            kind = cell(pr, pc)
            if kind == 3: return car.index(), -1.0, True
            if kind == 0:
                car.to_start(STARTS[np.random.randint(len(STARTS))])
                return car.index(), -1.0, False
        car.r, car.c = point(car.r, car.c, car.vy, car.vx, n, n)
        return car.index(), -1.0, False


@jitclass([('me', Car.class_type.instance_type), ('other', Car.class_type.instance_type),
           ('policy', int64[:]), ('noise', float64), ('other_noise', float64), ('radius', int64), ('encoding', int64), ('crashes', int64),
           ('collisions', int64), ('steps', int64)])
class TwoCarEnv:
    """The learner plus an opponent that drives a fixed policy (the greedy Part 1 policy).
    Both cars move at once; if they come within `radius` cells (on both axes) at the same point
    along their moves, the learner is sent back to the start line. The opponent restarts when it finishes or crashes.
    encoding 0: own state only; 1: + other car's offset, clipped to a 7x7 window (or 'far');
    2: + the sign of the velocity difference on each axis."""

    def __init__(self, policy, noise, other_noise, radius, encoding):
        self.me = Car(); self.other = Car(); self.policy = policy
        self.noise = noise; self.other_noise = other_noise; self.radius = radius; self.encoding = encoding
        self.crashes = 0; self.collisions = 0; self.steps = 0

    def n_obs(self):
        return N_POS * (1, 50, 450)[self.encoding]

    def _respawn(self, car, avoid):
        while True:
            col = STARTS[np.random.randint(len(STARTS))]
            if col != avoid: break
        car.to_start(col)

    def obs(self):
        o = self.me.index()
        if self.encoding == 0: return o
        dy, dx = self.other.r - self.me.r, self.other.c - self.me.c
        rel = (dy + 3) * 7 + (dx + 3) if abs(dy) <= 3 and abs(dx) <= 3 else 49
        if self.encoding == 1: return o * 50 + rel
        sy = int(np.sign(self.other.vy - self.me.vy)) + 1
        sx = int(np.sign(self.other.vx - self.me.vx)) + 1
        return (o * 50 + rel) * 9 + sy * 3 + sx

    def reset(self):
        return self.reset_at(STARTS[np.random.randint(len(STARTS))])

    def reset_at(self, col):
        self.me.to_start(col)
        self._respawn(self.other, self.me.c)
        self.crashes = 0; self.collisions = 0; self.steps = 0
        return self.obs()

    def legal(self, a):
        return legal(self.me.vy, self.me.vx, a)

    def step(self, a):
        me, ot = self.me, self.other; self.steps += 1
        me.accelerate(a, self.noise)
        ot.accelerate(self.policy[ot.index()], self.other_noise)
        n = max(me.vy, me.vx, ot.vy, ot.vx)
        if n == 0: return self.obs(), -1.0, False
        me_alive = ot_alive = True
        for k in range(1, n + 1):
            mr, mc = point(me.r, me.c, me.vy, me.vx, k, n)
            orr, oc = point(ot.r, ot.c, ot.vy, ot.vx, k, n)
            if ot_alive:
                kind = cell(orr, oc)
                if kind != 1:                                  # opponent finished or crashed
                    self._respawn(ot, -1); ot_alive = False
            kind = cell(mr, mc)
            if kind == 3: return self.obs(), -1.0, True
            if kind == 0:
                self.crashes += 1; self._respawn(me, -1)
                me_alive = False; break
            if ot_alive and abs(mr - orr) <= self.radius and abs(mc - oc) <= self.radius:
                self.collisions += 1; self._respawn(me, -1); me_alive = False; break
        if me_alive: me.r, me.c = point(me.r, me.c, me.vy, me.vx, n, n)
        if ot_alive: ot.r, ot.c = point(ot.r, ot.c, ot.vy, ot.vx, n, n)
        return self.obs(), -1.0, False


@jitclass([('Q', float32[:, :]), ('C', float32[:, :]), ('eps', float64)])
class OffPolicyMC:
    """Off-policy MC control with weighted importance sampling (Section 5.7).
    Behaviour: epsilon-greedy over legal actions. Target: greedy (ties to the lowest index).
    Q starts at -1000, so an action the agent has never tried never looks best."""

    def __init__(self, n_obs, eps):
        self.Q = np.full((n_obs, 9), -1000, np.float32); self.C = np.zeros((n_obs, 9), np.float32)
        self.eps = eps

    def greedy(self, s, mask):
        best, pick = -np.inf, -1
        for a in range(9):
            if mask[a] and self.Q[s, a] > best: best, pick = self.Q[s, a], a
        return pick

    def behave(self, s, mask):
        """Returns (action, probability the behaviour policy gave it)."""
        n = mask.sum(); g = self.greedy(s, mask)
        if np.random.random() < self.eps:
            k = np.random.randint(n)
            for a in range(9):
                if mask[a]:
                    if k == 0: break
                    k -= 1
        else:
            a = g
        return a, self.eps / n + (1 - self.eps) * (a == g)

    def update(self, S, A, R, P, M, T):
        """Backward pass over one episode; M holds the legal masks seen at each step."""
        G, W = 0.0, 1.0
        for t in range(T - 1, -1, -1):
            s, a = S[t], A[t]
            G += R[t]
            self.C[s, a] += W
            self.Q[s, a] += W / self.C[s, a] * (G - self.Q[s, a])
            if a != self.greedy(s, M[t]): break
            W /= P[t]


@njit(cache=False)
def masks(env):
    m = np.zeros(9, np.bool_)
    for a in range(9): m[a] = env.legal(a)
    return m


@njit(cache=False)
def episode(env, agent, cap, S, A, R, P, M):
    s = env.reset(); T = 0
    while T < cap:
        m = masks(env); a, p = agent.behave(s, m)
        s2, r, done = env.step(a)
        S[T] = s; A[T] = a; R[T] = r; P[T] = p; M[T] = m; T += 1
        s = s2
        if done: break
    agent.update(S, A, R, P, M, T)
    return T


@njit(cache=False)
def greedy_run(env, agent, cap, start_col):
    """Steps for the greedy policy to finish from a start cell (cap if it never does)."""
    s = env.reset_at(start_col)
    for t in range(cap):
        s, r, done = env.step(agent.greedy(s, masks(env)))
        if done: return t + 1
    return cap


def make_policy(agent):
    """Greedy action for every own-state (for driving the opponent car in Part 2)."""
    pol = np.zeros(N_POS, np.int64)
    for i in range(N_POS):
        vx = i % (VMAX + 1); vy = (i // (VMAX + 1)) % (VMAX + 1)
        m = np.array([legal(vy, vx, a) for a in range(9)])
        q = np.where(m, agent.Q[i], -np.inf)
        pol[i] = int(np.argmax(q))
    return pol
