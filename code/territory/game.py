"""Part 3 rules: the 5x5 territory game, ported tick for tick from play/territory.html.

A state is (land bitmask, trail tuple, player cell, heading, hunter cell). Heading 4 = not moving yet.
step() returns every possible next state; there is more than one only when the hunter respawns."""
N = 5
M = N * N
DX = (0, 0, -1, 1)                    # up, down, left, right (same order as the page)
DY = (-1, 1, 0, 0)
REV = (1, 0, 3, 2)
WAIT = 4
NB = [[(k, (c // N + DY[k]) * N + c % N + DX[k]) for k in range(4)
       if 0 <= c % N + DX[k] < N and 0 <= c // N + DY[k] < N] for c in range(M)]
BORDER = [c for c in range(M) if c % N in (0, N - 1) or c // N in (0, N - 1)]
START = (1 << (M // 2), (), M // 2, WAIT, 0)    # one home square in the centre, hunter in a corner


def area(s):
    return bin(s[0]).count('1')


def field(own, path, me):
    """Walking distance (around land) to the hunter's goal: the trail, or while the player is
    home, the open cells closest to the player in straight-line distance."""
    D = [-1] * M
    if path:
        q = list(path)
    else:
        mx, my = me % N, me // N
        d2 = {c: (c % N - mx) ** 2 + (c // N - my) ** 2 for c in range(M) if not own >> c & 1}
        q = [c for c in d2 if d2[c] == min(d2.values())]
    for c in q: D[c] = 0
    for c in q:                                      # q grows while we iterate: a BFS
        for _, n in NB[c]:
            if D[n] < 0 and not own >> n & 1:
                D[n] = D[c] + 1; q.append(n)
    return D


def move_hunter(f, D, me):
    """One step closer to the goal; ties go to the move that closes the larger gap to the player."""
    here = D[f]
    if here <= 0: return f
    tx, ty = me % N - f % N, me // N - f // N
    best, score = f, -float('inf')
    for k, n in NB[f]:
        if 0 <= D[n] < here:
            sc = DX[k] * tx * abs(tx) + DY[k] * ty * abs(ty)
            if sc > score: best, score = n, sc
    return best


def capture(own, path):
    """The trail becomes land, then every open cell that cannot reach the board edge is filled."""
    for c in path: own |= 1 << c
    seen, st = 0, [c for c in BORDER if not own >> c & 1]
    for c in st: seen |= 1 << c
    while st:
        c = st.pop()
        for _, n in NB[c]:
            if not own >> n & 1 and not seen >> n & 1:
                seen |= 1 << n; st.append(n)
    return own | ((1 << M) - 1) & ~seen


def actions(s):
    """Distinct choices: before the first move the player may also wait; reversing onto a trail
    is ignored by the game, so it is the same as continuing."""
    if s[3] == WAIT: return [WAIT, 0, 1, 2, 3]
    return [k for k in range(4) if not (s[1] and k == REV[s[3]])]


def step(s, a):
    """One tick. Returns a list of outcomes: ('dead', land) or a next state."""
    own, path, me, d, f = s
    if a != WAIT and not (d != WAIT and path and a == REV[d]): d = a
    if d != WAIT:
        x, y = me % N + DX[d], me // N + DY[d]
        if not (0 <= x < N and 0 <= y < N): return [('dead', own)]       # hit the wall
        me = y * N + x
        if me in path: return [('dead', own)]                              # crossed own trail
        if not own >> me & 1: path = path + (me,)
    if f not in path: f = move_hunter(f, field(own, path, me), me)
    if f in path: return [('dead', own)]                                   # hunter reached the trail
    if own >> me & 1 and path:
        own = capture(own, path); path = ()
        if own >> f & 1:                                                   # hunter enclosed: respawn anywhere open
            return [(own, (), me, d, c) for c in range(M) if not own >> c & 1] or [(own, (), me, d, 0)]
    return [(own, path, me, d, f)]
