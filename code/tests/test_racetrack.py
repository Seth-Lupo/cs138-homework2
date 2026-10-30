import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from racetrack import ACC, ROWS, STARTS, OffPolicyMC, RaceEnv, TwoCarEnv, N_POS, legal


def action(ay, ax):
    return int(np.flatnonzero((ACC[:, 0] == ay) & (ACC[:, 1] == ax))[0])


class RaceTests(unittest.TestCase):
    def test_velocity_rules(self):
        self.assertFalse(legal(0, 0, action(0, 0)))         # never (0, 0) after an action
        self.assertTrue(legal(0, 0, action(1, 0)))
        self.assertFalse(legal(4, 0, action(1, 0)))         # at most 4
        self.assertFalse(legal(0, 1, action(0, -1)))        # never negative or (0, 0)

    def test_crash_returns_to_start(self):
        env = RaceEnv(0.0); env.reset_at(3)
        env.step(action(0, 1)); env.step(action(0, 1))      # right into the wall at column 10
        env.step(action(0, 1)); env.step(action(0, 1))
        self.assertEqual((env.car.r, env.car.vy, env.car.vx), (ROWS - 1, 0, 0))

    def test_finish_line(self):
        env = RaceEnv(0.0); env.reset_at(3)
        env.car.r, env.car.c, env.car.vy, env.car.vx = 3, 12, 0, 3
        _, r, done = env.step(action(0, 0))
        self.assertTrue(done); self.assertEqual(r, -1.0)

    def test_noise_skips_acceleration(self):
        env = RaceEnv(1.0); env.reset_at(5)
        env.step(action(1, 1))
        self.assertEqual((env.car.vy, env.car.vx), (0, 0))

    def test_collision_resets_learner(self):
        policy = np.full(N_POS, action(1, 0), np.int64)    # opponent always accelerates upward
        for radius, hits in ((0, 0), (1, 1)):
            env = TwoCarEnv(policy, 0.0, 0.0, radius, 1); env.reset_at(4)
            env.other.to_start(4); env.other.r = ROWS - 2     # opponent one cell ahead
            env.step(action(1, 0))                            # both move up one: one cell apart
            self.assertEqual(env.collisions, hits)
        self.assertEqual((env.me.r, env.me.vy, env.me.vx), (ROWS - 1, 0, 0))

    def test_weighted_importance_sampling(self):
        agent = OffPolicyMC(4, 0.1)
        S = np.array([0, 1]); A = np.array([0, 0]); R = np.array([-1.0, -1.0])
        P = np.array([0.5, 0.5]); M = np.ones((2, 9), np.bool_)
        agent.Q[0, 0] = agent.Q[1, 0] = 0                  # make action 0 greedy in both states
        agent.update(S, A, R, P, M, 2)
        self.assertAlmostEqual(agent.Q[1, 0], -1.0)          # first update copies the return
        self.assertAlmostEqual(agent.Q[0, 0], -2.0)


if __name__ == '__main__':
    unittest.main()
