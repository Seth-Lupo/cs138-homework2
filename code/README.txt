CS 138 Homework 2: Monte Carlo control on racetracks and a territory game

Python 3.11 or later. Run these commands from the project folder
(the parent of code/, containing main.tex and results/):

  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r code/requirements.txt
  python -m unittest discover -s code/tests -v
  python code/run.py              # all experiments (about 15 minutes on 8 cores)
  python code/figures.py          # results/figures/*.pdf
  pdflatex main.tex               # the report

run.py --part 1|2|3 runs one part. Every run is seeded from 138.

Files
  racetrack.py       Parts 1-2. Track (Figure 5.5, left), RaceEnv, TwoCarEnv and the
                     off-policy MC agent (weighted importance sampling), as Numba jitclasses.
  territory/game.py  Part 3 rules, ported tick for tick from play/territory.html.
  territory/solve.py Enumerates all reachable states, computes the exact optimum, and writes
                     the transition table the MC environment steps through.
  territory/mc.py    TerritoryEnv and MCControl (on-policy epsilon-soft or exploring starts).
  play/territory.html  The playable game, with a replay of the optimal line.
  tests/             Racetrack physics and update rule; territory rules, the optimal line,
                     and a tick-for-tick match against the page (needs node; skipped otherwise).
  prompts/           Code-generation prompts, one file per part.

Agents only see an observation index, the reward and the legal actions; environments never
expose their internals to the agent.

Results (results/)
  part1/curves.npz   greedy steps [eps, run, checkpoint], behaviour episode length, the
                     opponent policy for Part 2, and the noise-free optimum per start cell
  part2/curves.npz   greedy steps and collisions [radius, encoding, run, checkpoint]
  part3/table.npz    exact transition table and values; part3/curves.npz greedy land and
                     states visited [method, run, checkpoint]; games.json final greedy games
  */config.json      the settings used
