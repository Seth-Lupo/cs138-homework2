# CS 138 Homework 2

Monte Carlo control on the racetrack (Sutton and Barto, Exercise 5.12), a two-car version of it, and a 5×5 territory game that is solved exactly and then learned with Monte Carlo.

## Code and prompts

- `code/`: environments, agents, the territory-game solver, experiment runner, figures, and tests.
- `code/play/territory.html`: the playable game ([online](https://claude.ai/artifact/C4rPs6KtsxpaXBTdWHnDiw)).
- `code/prompts/`: code-generation prompts.
- `code/README.txt`: file list and reproduction commands. References there to the report refer to the accompanying homework package.

## Quick start

Run from the repository root with Python 3.11 or later:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
python -m unittest discover -s code/tests -v
python code/run.py          # all three parts, about 15 minutes on 8 cores
python code/figures.py      # results/figures/*.pdf
```

`python code/run.py --part 1|2|3` runs one part. The territory-page test needs Node.js and is skipped without it.
