"""Utilities for parsing match fixtures and running Monte Carlo simulations.

This module reads fixtures exported from SportsClubStats and provides
functions to compute league tables and run a SportsClubStats-style model to
project results.  It powers the public functions exposed in
``__init__``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict

from joblib import Parallel, delayed

try:  # Optional dependency for progress bars
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm is optional in tests
    tqdm = None

import numpy as np
import pandas as pd
import math
import functools

try:  # Optional dependency for fast Poisson quantile
    from scipy.stats import poisson as _scipy_poisson  # type: ignore
except Exception:  # pragma: no cover - SciPy is optional
    _scipy_poisson = None

# ---------------------------------------------------------------------------
# Default simulation parameters
# ---------------------------------------------------------------------------

# Percent chance of a match ending in a tie when teams are of equal strength
# Fixed at one third of matches ending level.
DEFAULT_TIE_PERCENT = 33.3

# Relative advantage multiplier for the home team
# No built-in home advantage is used by default.
DEFAULT_HOME_FIELD_ADVANTAGE = 1.0

# Default number of parallel jobs. Use all available cores.
DEFAULT_JOBS = os.cpu_count() or 1

# Number of simulations to execute per parallel batch when running with
# ``n_jobs`` greater than one. Smaller batches keep memory usage low while
# ensuring deterministic ordering of results.
_BATCH_SIZE = 100



# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------
SCORE_PATTERN = re.compile(
    r"(\d+/\d+/\d+)\s+(.+?)\s+(\d+)-(\d+)\s+(.+?)\s*(?:\(ID:.*)?$"
)
NOSCORE_PATTERN = re.compile(
    r"(\d+/\d+/\d+)\s+(.+?)\s{2,}(.+?)\s*(?:\(ID:.*)?$"
)


def _parse_date(date_str: str) -> pd.Timestamp:
    parts = date_str.split("/")
    year = parts[-1]
    if len(year) == 4:
        return pd.to_datetime(date_str, format="%d/%m/%Y")
    return pd.to_datetime(date_str, format="%m/%d/%y")


def parse_matches(path: str | Path) -> pd.DataFrame:
    """Return a DataFrame of fixtures and results."""
    rows: list[dict] = []
    in_games = False
    saw_begin = False
    saw_end = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "GamesBegin":
                saw_begin = True
                in_games = True
                continue
            if stripped == "GamesEnd":
                saw_end = True
                break
            if not in_games:
                continue
            line = line.rstrip("\n")
            # Skip known metadata lines that may appear within the games section
            if not line or line.startswith("TeamListedFirst:"):
                continue
            m = SCORE_PATTERN.match(line)
            if m:
                date_str, home, hs, as_, away = m.groups()
                rows.append(
                    {
                        "date": _parse_date(date_str),
                        "home_team": home.strip(),
                        "away_team": away.strip(),
                        "home_score": int(hs),
                        "away_score": int(as_),
                    }
                )
                continue
            m = NOSCORE_PATTERN.match(line)
            if m:
                date_str, home, away = m.groups()
                rows.append(
                    {
                        "date": _parse_date(date_str),
                        "home_team": home.strip(),
                        "away_team": away.strip(),
                        "home_score": np.nan,
                        "away_score": np.nan,
                    }
                )
                continue
            # Any line within the Games section that doesn't match one of the
            # expected patterns should trigger an error to aid debugging.
            raise ValueError(f"Unrecognized line in matches file: {line}")
    if not (saw_begin and saw_end):
        raise ValueError("Matches file must contain 'GamesBegin' and 'GamesEnd' markers")
    return pd.DataFrame(rows)


def reset_results_from(matches: pd.DataFrame, start_date: str | pd.Timestamp) -> pd.DataFrame:
    """Return a copy of ``matches`` with results on or after ``start_date`` cleared."""

    df = matches.copy()
    start = pd.to_datetime(start_date)
    mask = df["date"] >= start
    df.loc[mask, ["home_score", "away_score"]] = np.nan
    return df


# ---------------------------------------------------------------------------
# Table computation
# ---------------------------------------------------------------------------


def _head_to_head_points(
    matches: pd.DataFrame,
    teams: list[str],
) -> Dict[str, int]:
    points = {t: 0 for t in teams}
    df = matches.dropna(subset=["home_score", "away_score"])
    df = df[df["home_team"].isin(teams) & df["away_team"].isin(teams)]
    for _, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        if hs > as_:
            points[ht] += 3
        elif hs < as_:
            points[at] += 3
        else:
            points[ht] += 1
            points[at] += 1
    return points


def league_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute league standings from played matches."""
    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    played = matches.dropna(subset=["home_score", "away_score"])

    if played.empty:
        df = pd.DataFrame(
            {
                "team": teams,
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "gf": 0,
                "ga": 0,
                "points": 0,
                "gd": 0,
            }
        )
    else:
        home = played[["home_team", "home_score", "away_score"]].rename(
            columns={"home_team": "team", "home_score": "gf", "away_score": "ga"}
        )
        away = played[["away_team", "away_score", "home_score"]].rename(
            columns={"away_team": "team", "away_score": "gf", "home_score": "ga"}
        )
        for df_part in (home, away):
            df_part["wins"] = (df_part["gf"] > df_part["ga"]).astype(int)
            df_part["draws"] = (df_part["gf"] == df_part["ga"]).astype(int)
            df_part["losses"] = (df_part["gf"] < df_part["ga"]).astype(int)

        stats = (
            pd.concat([home, away], ignore_index=True)
            .groupby("team", sort=False)
            .agg(
                played=("wins", "size"),
                wins=("wins", "sum"),
                draws=("draws", "sum"),
                losses=("losses", "sum"),
                gf=("gf", "sum"),
                ga=("ga", "sum"),
            )
        ).astype(int)

        stats["points"] = stats["wins"] * 3 + stats["draws"]
        stats["gd"] = stats["gf"] - stats["ga"]
        df = stats.reindex(teams, fill_value=0).reset_index().rename(columns={"index": "team"})
    df["head_to_head"] = 0
    for _, group in df.groupby(["points", "wins", "gd", "gf"]):
        if len(group) <= 1:
            continue
        teams_tied = group["team"].tolist()
        h2h = _head_to_head_points(played, teams_tied)
        for t, val in h2h.items():
            df.loc[df["team"] == t, "head_to_head"] = val

    df = df.sort_values(
        ["points", "wins", "gd", "gf", "head_to_head", "team"],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def _poisson_ppf(u: float, lam: float) -> int:
    """Return the Poisson quantile for ``u`` with mean ``lam``.

    When SciPy is available this delegates to ``scipy.stats.poisson.ppf``
    which uses a highly optimised implementation. A NumPy based fallback
    relying on log-probabilities and binary search is provided when SciPy
    is unavailable.
    """

    if _scipy_poisson is not None:
        return int(_scipy_poisson.ppf(u, lam))

    if lam <= 0:
        return 0

    cdf = _poisson_cdf_array(float(lam))
    idx = int(np.searchsorted(cdf, u, side="left"))
    return idx


@functools.lru_cache(maxsize=None)
def _poisson_cdf_array(lam: float) -> np.ndarray:
    """Pre-compute the Poisson CDF values for ``lam``.

    The array contains cumulative probabilities up to ``lam + 10 * sqrt(lam)``
    which effectively captures the upper tail. Results are cached to avoid
    recomputation when ``_poisson_ppf`` is called repeatedly with the same
    ``lam``.
    """

    max_k = int(np.ceil(lam + 10 * math.sqrt(lam)))
    ks = np.arange(0, max_k + 1)
    if max_k > 0:
        log_fact = np.concatenate(([0.0], np.cumsum(np.log(ks[1:], dtype=float))))
    else:
        log_fact = np.array([0.0])
    log_pmf = ks * math.log(lam) - lam - log_fact
    return np.cumsum(np.exp(log_pmf))


def _simulate_table(
    played_df: pd.DataFrame,
    remaining: pd.DataFrame,
    rng: np.random.Generator,
    *,
    tie_prob: float = DEFAULT_TIE_PERCENT / 100.0,
    home_advantage: float = DEFAULT_HOME_FIELD_ADVANTAGE,
    team_params: Dict[str, tuple[float, float]] | None = None,
    home_goals_mean: float | None = None,
    away_goals_mean: float | None = None,
    rho: float | None = None,
) -> pd.DataFrame:
    """Simulate remaining fixtures.

    When both ``home_goals_mean`` and ``away_goals_mean`` are provided the match
    scores are sampled from Poisson distributions with the given means scaled by
    ``home_advantage`` and any ``team_params``. Passing only one of the means
    is invalid. If ``rho`` is specified the home and away goals are drawn from
    correlated Poisson variates. Otherwise the classic win/draw/loss model based
    on ``tie_prob`` is used.
    """

    if not 0.0 <= tie_prob <= 1.0:
        raise ValueError("tie_prob must be between 0 and 1")
    if home_advantage <= 0:
        raise ValueError("home_advantage must be greater than zero")

    if home_goals_mean is not None and home_goals_mean <= 0:
        raise ValueError("home_goals_mean must be greater than zero")
    if away_goals_mean is not None and away_goals_mean <= 0:
        raise ValueError("away_goals_mean must be greater than zero")
    if (home_goals_mean is None) != (away_goals_mean is None):
        raise ValueError(
            "home_goals_mean and away_goals_mean must both be provided"
        )
    if rho is not None and (rho <= -1 or rho >= 1):
        raise ValueError("rho must be between -1 and 1")
    if rho is not None and home_goals_mean is None:
        raise ValueError("rho requires goal-based simulation")

    sims: list[dict] = []

    for _, row in remaining.iterrows():
        tp = tie_prob
        ha = home_advantage
        if team_params is not None:
            home_att, home_def = team_params.get(row["home_team"], (1.0, 1.0))
            away_att, away_def = team_params.get(row["away_team"], (1.0, 1.0))
            ha *= home_att / away_def
            away_factor = away_att / home_def
        else:
            away_factor = 1.0

        if home_goals_mean is not None or away_goals_mean is not None:
            hm = home_goals_mean if home_goals_mean is not None else 1.0
            am = away_goals_mean if away_goals_mean is not None else 1.0
            hm *= ha
            am *= away_factor
            if rho is not None:
                z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]])
                u1 = 0.5 * (1.0 + math.erf(z[0] / math.sqrt(2.0)))
                u2 = 0.5 * (1.0 + math.erf(z[1] / math.sqrt(2.0)))
                hs = _poisson_ppf(u1, hm)
                as_ = _poisson_ppf(u2, am)
            else:
                hs = int(rng.poisson(hm))
                as_ = int(rng.poisson(am))
        else:
            rest = 1.0 - tp
            strength_sum = ha + away_factor
            home_prob = rest * ha / strength_sum
            draw_prob = tp
            r = rng.random()
            if r < home_prob:
                hs, as_ = 1, 0
            elif r < home_prob + draw_prob:
                hs, as_ = 0, 0
            else:
                hs, as_ = 0, 1

        sims.append(
            {
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_score": hs,
                "away_score": as_,
            }
        )
    all_matches = pd.concat([played_df, pd.DataFrame(sims)], ignore_index=True)
    return league_table(all_matches)


def _iterate_tables(
    played_df: pd.DataFrame,
    remaining: pd.DataFrame,
    rng: np.random.Generator,
    iterations: int,
    *,
    desc: str,
    progress: bool,
    tie_prob: float,
    home_advantage: float,
    team_params: Dict[str, tuple[float, float]] | None,
    home_goals_mean: float | None,
    away_goals_mean: float | None,
    rho: float | None,
    n_jobs: int,
):
    """Yield successive simulated tables.

    This helper mirrors the parallel and serial execution logic used by the
    public simulation functions to ensure deterministic ordering of results.
    """

    if n_jobs == 1:
        iterator = range(iterations)
        if progress and tqdm is not None:
            iterator = tqdm(iterator, desc=desc, unit="sim")
        for _ in iterator:
            yield _simulate_table(
                played_df,
                remaining,
                rng,
                tie_prob=tie_prob,
                home_advantage=home_advantage,
                team_params=team_params,
                home_goals_mean=home_goals_mean,
                away_goals_mean=away_goals_mean,
                rho=rho,
            )
    else:
        seeds = rng.integers(0, 2**32 - 1, size=iterations)
        pbar = None
        if progress and tqdm is not None:
            pbar = tqdm(seeds, desc=desc, unit="sim")

        def run(seed: int) -> pd.DataFrame:
            return _simulate_table(
                played_df,
                remaining,
                np.random.default_rng(seed),
                tie_prob=tie_prob,
                home_advantage=home_advantage,
                team_params=team_params,
                home_goals_mean=home_goals_mean,
                away_goals_mean=away_goals_mean,
                rho=rho,
            )

        for start in range(0, iterations, _BATCH_SIZE):
            batch = seeds[start : start + _BATCH_SIZE]
            results = Parallel(n_jobs=n_jobs)(delayed(run)(s) for s in batch)
            if pbar is not None and hasattr(pbar, "update"):
                pbar.update(len(batch))
            for table in results:
                yield table
        if pbar is not None and hasattr(pbar, "close"):
            pbar.close()


# ---------------------------------------------------------------------------
# Public simulation API
# ---------------------------------------------------------------------------


def simulate_chances(
    matches: pd.DataFrame,
    iterations: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    progress: bool = True,
    tie_prob: float = DEFAULT_TIE_PERCENT / 100.0,
    home_advantage: float = DEFAULT_HOME_FIELD_ADVANTAGE,
    team_params: Dict[str, tuple[float, float]] | None = None,
    home_goals_mean: float | None = None,
    away_goals_mean: float | None = None,
    rho: float | None = None,
    n_jobs: int = DEFAULT_JOBS,
) -> Dict[str, float]:
    """Return title probabilities.

    If goal means are provided, both ``home_goals_mean`` and ``away_goals_mean``
    must be supplied to enable Poisson-based scoring.
    """

    if n_jobs <= 0:
        raise ValueError("n_jobs must be greater than 0")

    if iterations <= 0:
        raise ValueError("iterations must be greater than 0")

    if rng is None:
        rng = np.random.default_rng()


    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    champs = {t: 0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[
        matches["home_score"].isna() | matches["away_score"].isna()
    ]

    for table in _iterate_tables(
        played_df,
        remaining,
        rng,
        iterations,
        desc="Chances",
        progress=progress,
        tie_prob=tie_prob,
        home_advantage=home_advantage,
        team_params=team_params,
        home_goals_mean=home_goals_mean,
        away_goals_mean=away_goals_mean,
        rho=rho,
        n_jobs=n_jobs,
    ):
        champs[table.iloc[0]["team"]] += 1

    for t in champs:
        champs[t] /= iterations
    return champs


def simulate_relegation_chances(
    matches: pd.DataFrame,
    iterations: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    progress: bool = True,
    tie_prob: float = DEFAULT_TIE_PERCENT / 100.0,
    home_advantage: float = DEFAULT_HOME_FIELD_ADVANTAGE,
    team_params: Dict[str, tuple[float, float]] | None = None,
    home_goals_mean: float | None = None,
    away_goals_mean: float | None = None,
    rho: float | None = None,
    n_jobs: int = DEFAULT_JOBS,
) -> Dict[str, float]:
    """Return probabilities of finishing in the bottom four.

    If goal means are provided, both ``home_goals_mean`` and ``away_goals_mean``
    must be supplied to enable Poisson-based scoring.
    """

    if n_jobs <= 0:
        raise ValueError("n_jobs must be greater than 0")

    if iterations <= 0:
        raise ValueError("iterations must be greater than 0")

    if rng is None:
        rng = np.random.default_rng()


    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    relegated = {t: 0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[
        matches["home_score"].isna() | matches["away_score"].isna()
    ]
    for table in _iterate_tables(
        played_df,
        remaining,
        rng,
        iterations,
        desc="Relegation",
        progress=progress,
        tie_prob=tie_prob,
        home_advantage=home_advantage,
        team_params=team_params,
        home_goals_mean=home_goals_mean,
        away_goals_mean=away_goals_mean,
        rho=rho,
        n_jobs=n_jobs,
    ):
        for team in table.tail(4)["team"]:
            relegated[team] += 1

    for t in relegated:
        relegated[t] /= iterations
    return relegated


def simulate_final_table(
    matches: pd.DataFrame,
    iterations: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    progress: bool = True,
    tie_prob: float = DEFAULT_TIE_PERCENT / 100.0,
    home_advantage: float = DEFAULT_HOME_FIELD_ADVANTAGE,
    team_params: Dict[str, tuple[float, float]] | None = None,
    home_goals_mean: float | None = None,
    away_goals_mean: float | None = None,
    rho: float | None = None,
    n_jobs: int = DEFAULT_JOBS,
) -> pd.DataFrame:
    """Project average finishing position and points.

    If goal means are provided, both ``home_goals_mean`` and ``away_goals_mean``
    must be supplied to enable Poisson-based scoring.
    """

    if n_jobs <= 0:
        raise ValueError("n_jobs must be greater than 0")

    if iterations <= 0:
        raise ValueError("iterations must be greater than 0")

    if rng is None:
        rng = np.random.default_rng()


    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    pos_totals = {t: 0.0 for t in teams}
    points_totals = {t: 0.0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[
        matches["home_score"].isna() | matches["away_score"].isna()
    ]

    for table in _iterate_tables(
        played_df,
        remaining,
        rng,
        iterations,
        desc="Final table",
        progress=progress,
        tie_prob=tie_prob,
        home_advantage=home_advantage,
        team_params=team_params,
        home_goals_mean=home_goals_mean,
        away_goals_mean=away_goals_mean,
        rho=rho,
        n_jobs=n_jobs,
    ):
        for idx, row in table.iterrows():
            pos_totals[row["team"]] += idx + 1
            points_totals[row["team"]] += row["points"]

    results = []
    for team in teams:
        results.append(
            {
                "team": team,
                "position": pos_totals[team] / iterations,
                "points": points_totals[team] / iterations,
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values("position").reset_index(drop=True)
    return df


def summary_table(
    matches: pd.DataFrame,
    iterations: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    progress: bool = True,
    tie_prob: float = DEFAULT_TIE_PERCENT / 100.0,
    home_advantage: float = DEFAULT_HOME_FIELD_ADVANTAGE,
    team_params: Dict[str, tuple[float, float]] | None = None,
    home_goals_mean: float | None = None,
    away_goals_mean: float | None = None,
    rho: float | None = None,
    n_jobs: int = DEFAULT_JOBS,
) -> pd.DataFrame:
    """Return a combined projection table ranked by expected points.

    The ``position`` column corresponds to the rank after sorting by the
    expected point totals. If goal means are provided, both ``home_goals_mean``
    and ``away_goals_mean`` must be supplied to enable Poisson-based scoring.
    """

    if n_jobs <= 0:
        raise ValueError("n_jobs must be greater than 0")

    if iterations <= 0:
        raise ValueError("iterations must be greater than 0")

    if rng is None:
        rng = np.random.default_rng()


    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    title_counts = {t: 0 for t in teams}
    top4_counts = {t: 0 for t in teams}
    relegated = {t: 0 for t in teams}
    points_totals = {t: 0.0 for t in teams}
    wins_totals = {t: 0.0 for t in teams}
    gd_totals = {t: 0.0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[
        matches["home_score"].isna() | matches["away_score"].isna()
    ]

    for table in _iterate_tables(
        played_df,
        remaining,
        rng,
        iterations,
        desc="Summary",
        progress=progress,
        tie_prob=tie_prob,
        home_advantage=home_advantage,
        team_params=team_params,
        home_goals_mean=home_goals_mean,
        away_goals_mean=away_goals_mean,
        rho=rho,
        n_jobs=n_jobs,
    ):
        title_counts[table.iloc[0]["team"]] += 1
        for team in table.head(4)["team"]:
            top4_counts[team] += 1
        for team in table.tail(4)["team"]:
            relegated[team] += 1
        for _, row in table.iterrows():
            points_totals[row["team"]] += row["points"]
            wins_totals[row["team"]] += row["wins"]
            gd_totals[row["team"]] += row["gd"]

    rows = []
    for team in teams:
        rows.append(
            {
                "team": team,
                "points": points_totals[team] / iterations,
                "wins": wins_totals[team] / iterations,
                "gd": gd_totals[team] / iterations,
                "title": title_counts[team] / iterations,
                "top4": top4_counts[team] / iterations,
                "relegation": relegated[team] / iterations,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["points", "wins", "gd"], ascending=[False, False, False]).reset_index(drop=True)
    df["position"] = range(1, len(df) + 1)
    df["points"] = df["points"].round().astype(int)
    df["wins"] = df["wins"].round().astype(int)
    df["gd"] = df["gd"].round().astype(int)
    return df[["position", "team", "points", "wins", "gd", "title", "top4", "relegation"]]
