# Brasileirão Simulator

This project provides a simple simulator for the 2025 Brasileirão Série A season. It parses the fixtures provided in `data/Brasileirao2025A.txt`, builds a league table from played matches and simulates the remaining games many times to estimate title, top-four and relegation probabilities.

## Usage

Install dependencies from `requirements.txt` and run the simulator:

```bash
pip install -r requirements.txt
python main.py --auto-calibrate --simulations 1000
```

A progress bar is shown by default during the simulation runs. Use the
`--no-progress` flag to disable it when running just a few iterations or when
incorporating the simulator into automated scripts.

The simulator runs in parallel by default using all available CPU cores. Use the
`--jobs` option to specify a custom number of workers. Passing `--jobs 4` for
example will execute the Monte Carlo iterations using four parallel processes.
The summary table is automatically saved as `brasileirao.html` in the same
directory as `main.py`. Pass `--html-output <file>` to choose a custom path.
Use `--from-date YYYY-MM-DD` to ignore results on or after a given date and
simulate from that point forward. Use `--auto-calibrate` to derive the draw
percentage and home advantage from past seasons before running the simulation.
When this flag is supplied, every file matching `data/Brasileirao????A.txt` is
loaded automatically (except the file provided via `--file`).
Pass `--auto-team-strengths` to compute attack and defense multipliers for each
club from those same historical files and include them in the simulation.

The default draw rate and home-field advantage are
`DEFAULT_TIE_PERCENT` (33.3) and `DEFAULT_HOME_FIELD_ADVANTAGE` (1.0).
Use `--tie-percent` and `--home-advantage` to override these values on the
command line. `DEFAULT_JOBS` still defines the parallelism level.

Pass `--home-goals-mean` and `--away-goals-mean` to sample scores from Poisson
distributions with the given expected values instead of the basic win/draw/loss
model. These options can also be provided programmatically via the simulation
functions. Pass `--rho` to introduce correlation between the home and away
scorelines when using Poisson scoring.

Alternatively, pass `--auto-calibrate` to estimate these parameters using all
historical files in the `data/` directory. The computed draw rate and home
advantage are then used for the simulation. Combine this flag with
`--auto-team-strengths` to automatically incorporate team quality estimates.

Use ``estimate_team_strengths`` to calculate attack and defense multipliers for
each club.  Pass multiple seasons and an optional ``decay`` factor to weigh
recent results more heavily:

```python
from calibration import estimate_team_strengths
strengths = estimate_team_strengths([
    "data/Brasileirao2024A.txt",
    "data/Brasileirao2023A.txt",
], decay=0.9)
```

Pass ``strengths`` via the ``team_params`` argument when calling the simulation
functions to incorporate team quality into the projections.

By default matches are simulated purely at random with all teams considered
equal. When expected goals are supplied the scores are drawn from Poisson
distributions using those averages.

The script outputs the estimated chance of winning the title and of finishing in the top four for each team. It then prints the probability of each side finishing in the bottom four and being relegated. It also estimates the average final position and points of every club.
All of these metrics are derived from a single Monte Carlo loop so that title chances, relegation odds and projected points remain consistent.

## Tie-break Rules

When building the league table teams are ordered using the official Série A criteria:

1. Points
2. Number of wins
3. Goal difference
4. Goals scored
5. Points obtained in the games between the tied sides
6. Team name (alphabetical)

These rules are implemented in :func:`league_table` and therefore affect all simulation utilities.

## Project Layout

- `data/` – raw fixtures and results.
- `src/simulator.py` – parsing, table calculation and simulation routines.
- `main.py` – command-line interface to run the simulation.
- `tests/` – basic unit tests.

The main functions can be imported directly from the package:

```python
from simulator import (
    parse_matches,
    league_table,
    simulate_chances,
    simulate_relegation_chances,
    simulate_final_table,
    summary_table,
)
```

All simulation functions accept an optional ``n_jobs`` argument to control the
degree of parallelism. By default ``n_jobs`` is set to the number of CPU cores,
so simulations automatically run in parallel. When ``n_jobs`` is greater than
one, joblib is used to distribute the work across multiple workers. The tie
percentage and home advantage are fixed at their defaults of 33.3% and 1.0.
Provide expected goal values via ``home_goals_mean`` and ``away_goals_mean`` to
enable Poisson-based scoring. Add ``rho`` to correlate the home and away goal
processes when desired.

## License

This project is licensed under the [MIT License](LICENSE).
