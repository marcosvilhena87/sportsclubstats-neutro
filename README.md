# Brasileirão Simulator

This project provides a simple simulator for the 2025 Brasileirão Série A season. It parses the fixtures provided in `data/Brasileirao2025A.txt`, builds a league table from played matches and simulates the remaining games many times to estimate title and relegation probabilities.

## Usage

Install dependencies from `requirements.txt` and run the simulator:

```bash
pip install -r requirements.txt
python main.py --simulations 1000
```

A progress bar is shown by default during the simulation runs. Use the
`--no-progress` flag to disable it when running just a few iterations or when
incorporating the simulator into automated scripts.

The simulator runs in parallel by default using all available CPU cores. Use the
`--jobs` option to specify a custom number of workers. Passing `--jobs 4` for
example will execute the Monte Carlo iterations using four parallel processes.
The summary table is automatically saved as `brasileirao.html` in the same
directory as `main.py`. Pass `--html-output <file>` to choose a custom path.

The simulator also accepts `--tie-percent` and `--home-advantage` options to
control the share of matches ending in a draw and the bias towards the home
team. By default these values are kept fixed using the historical averages
`DEFAULT_TIE_PERCENT` (27.0) and `DEFAULT_HOME_FIELD_ADVANTAGE` (1.7).
Pass `--dynamic-params` to recalculate them from the games already played in
the data set. Per-team parameters are disabled by default. Use
`--per-team-params` to calculate a separate tie percentage and home advantage
for each club based on their home results. Per-team statistics can also be
updated after each simulated match so that probabilities adapt as the
simulation progresses. Enable this behaviour with `--update-team-params` or
leave it disabled for faster runs. `DEFAULT_JOBS` still defines the parallelism
level. The `--alpha` option controls the smoothing factor applied when updating
dynamic team parameters and must be between 0 and 1. The default value is 0.1.

Matches are simulated purely at random with all teams considered equal.

The script outputs the estimated chance of winning the title for each team. It then prints the probability of each side finishing in the bottom four and being relegated. It also estimates the average final position and points of every club.
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
- `src/brasileirao/simulator.py` – parsing, table calculation and simulation routines.
- `main.py` – command-line interface to run the simulation.
- `tests/` – basic unit tests.

The main functions can be imported directly from the package:

```python
from brasileirao import (
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
one, joblib is used to distribute the work across multiple workers. By default
the tie percentage and home advantage are kept fixed by default. Use
``--dynamic-params`` if you prefer to recalculate them from the supplied
``matches`` before each simulation run.

## License

This project is licensed under the [MIT License](LICENSE).
