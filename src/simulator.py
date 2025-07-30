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
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "GamesBegin":
                in_games = True
                continue
            if line.strip() == "GamesEnd":
                break
            if not in_games:
                continue
            line = line.rstrip("\n")
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
    table: Dict[str, Dict[str, float]] = {
        t: {
            "team": t,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "gf": 0,
            "ga": 0,
        }
        for t in teams
    }

    played = matches.dropna(subset=["home_score", "away_score"])
    for _, row in played.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        table[home]["played"] += 1
        table[away]["played"] += 1
        table[home]["gf"] += hs
        table[home]["ga"] += as_
        table[away]["gf"] += as_
        table[away]["ga"] += hs
        if hs > as_:
            table[home]["wins"] += 1
            table[home]["points"] = table[home].get("points", 0) + 3
            table[away]["losses"] += 1
            table[away].setdefault("points", 0)
        elif hs < as_:
            table[away]["wins"] += 1
            table[away]["points"] = table[away].get("points", 0) + 3
            table[home]["losses"] += 1
            table[home].setdefault("points", 0)
        else:
            table[home]["draws"] += 1
            table[away]["draws"] += 1
            table[home]["points"] = table[home].get("points", 0) + 1
            table[away]["points"] = table[away].get("points", 0) + 1

    for t in table.values():
        t.setdefault("points", 0)
        t["gd"] = t["gf"] - t["ga"]

    df = pd.DataFrame(table.values())
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

def _simulate_table(
    played_df: pd.DataFrame,
    remaining: pd.DataFrame,
    rng: np.random.Generator,
    *,
    tie_prob: float = DEFAULT_TIE_PERCENT / 100.0,
) -> pd.DataFrame:
    """Simulate remaining fixtures with fixed home advantage."""

    sims: list[dict] = []

    for _, row in remaining.iterrows():
        tp = tie_prob
        ha = DEFAULT_HOME_FIELD_ADVANTAGE
        rest = 1.0 - tp
        home_prob = rest * ha / (ha + 1)
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
    n_jobs: int = DEFAULT_JOBS,
) -> Dict[str, float]:
    """Return title probabilities.

    """

    if rng is None:
        rng = np.random.default_rng()


    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    champs = {t: 0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[
        matches["home_score"].isna() | matches["away_score"].isna()
    ]

    if n_jobs > 1:
        seeds = rng.integers(0, 2**32 - 1, size=iterations)
        iterator = seeds
        if progress and tqdm is not None:
            iterator = tqdm(iterator, desc="Chances", unit="sim")

        def run(seed: int) -> pd.DataFrame:
            return _simulate_table(
                played_df,
                remaining,
                np.random.default_rng(seed),
                tie_prob=tie_prob,
            )

        results = Parallel(n_jobs=n_jobs)(delayed(run)(s) for s in iterator)
        for table in results:
            champs[table.iloc[0]["team"]] += 1
    else:
        iterator = range(iterations)
        if progress and tqdm is not None:
            iterator = tqdm(iterator, desc="Chances", unit="sim")
        for _ in iterator:
            table = _simulate_table(
                played_df,
                remaining,
                rng,
                tie_prob=tie_prob,
            )
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
    n_jobs: int = DEFAULT_JOBS,
) -> Dict[str, float]:
    """Return probabilities of finishing in the bottom four."""

    if rng is None:
        rng = np.random.default_rng()


    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    relegated = {t: 0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[
        matches["home_score"].isna() | matches["away_score"].isna()
    ]
    if n_jobs > 1:
        seeds = rng.integers(0, 2**32 - 1, size=iterations)
        iterator = seeds
        if progress and tqdm is not None:
            iterator = tqdm(iterator, desc="Relegation", unit="sim")

        def run(seed: int) -> pd.DataFrame:
            return _simulate_table(
                played_df,
                remaining,
                np.random.default_rng(seed),
                tie_prob=tie_prob,
            )

        results = Parallel(n_jobs=n_jobs)(delayed(run)(s) for s in iterator)
        for table in results:
            for team in table.tail(4)["team"]:
                relegated[team] += 1
    else:
        iterator = range(iterations)
        if progress and tqdm is not None:
            iterator = tqdm(iterator, desc="Relegation", unit="sim")
        for _ in iterator:
            table = _simulate_table(
                played_df,
                remaining,
                rng,
                tie_prob=tie_prob,
            )
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
    n_jobs: int = DEFAULT_JOBS,
) -> pd.DataFrame:
    """Project average finishing position and points."""

    if rng is None:
        rng = np.random.default_rng()


    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    pos_totals = {t: 0.0 for t in teams}
    points_totals = {t: 0.0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[
        matches["home_score"].isna() | matches["away_score"].isna()
    ]

    if n_jobs > 1:
        seeds = rng.integers(0, 2**32 - 1, size=iterations)
        iterator = seeds
        if progress and tqdm is not None:
            iterator = tqdm(iterator, desc="Final table", unit="sim")

        def run(seed: int) -> pd.DataFrame:
            return _simulate_table(
                played_df,
                remaining,
                np.random.default_rng(seed),
                tie_prob=tie_prob,
            )

        results = Parallel(n_jobs=n_jobs)(delayed(run)(s) for s in iterator)
        for table in results:
            for idx, row in table.iterrows():
                pos_totals[row["team"]] += idx + 1
                points_totals[row["team"]] += row["points"]
    else:
        iterator = range(iterations)
        if progress and tqdm is not None:
            iterator = tqdm(iterator, desc="Final table", unit="sim")
        for _ in iterator:
            table = _simulate_table(
                played_df,
                remaining,
                rng,
                tie_prob=tie_prob,
            )
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
    n_jobs: int = DEFAULT_JOBS,
) -> pd.DataFrame:
    """Return a combined projection table ranked by expected points.

    The ``position`` column corresponds to the rank after sorting by the
    expected point totals.
    """

    if rng is None:
        rng = np.random.default_rng()


    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    title_counts = {t: 0 for t in teams}
    relegated = {t: 0 for t in teams}
    points_totals = {t: 0.0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[
        matches["home_score"].isna() | matches["away_score"].isna()
    ]

    if n_jobs > 1:
        seeds = rng.integers(0, 2**32 - 1, size=iterations)
        iterator = seeds
        if progress and tqdm is not None:
            iterator = tqdm(iterator, desc="Summary", unit="sim")

        def run(seed: int) -> pd.DataFrame:
            return _simulate_table(
                played_df,
                remaining,
                np.random.default_rng(seed),
                tie_prob=tie_prob,
            )

        results = Parallel(n_jobs=n_jobs)(delayed(run)(s) for s in iterator)
        for table in results:
            title_counts[table.iloc[0]["team"]] += 1
            for team in table.tail(4)["team"]:
                relegated[team] += 1
            for _, row in table.iterrows():
                points_totals[row["team"]] += row["points"]
    else:
        iterator = range(iterations)
        if progress and tqdm is not None:
            iterator = tqdm(iterator, desc="Summary", unit="sim")

        for _ in iterator:
            table = _simulate_table(
                played_df,
                remaining,
                rng,
                tie_prob=tie_prob,
            )
            title_counts[table.iloc[0]["team"]] += 1
            for team in table.tail(4)["team"]:
                relegated[team] += 1
            for _, row in table.iterrows():
                points_totals[row["team"]] += row["points"]

    rows = []
    for team in teams:
        rows.append(
            {
                "team": team,
                "points": points_totals[team] / iterations,
                "title": title_counts[team] / iterations,
                "relegation": relegated[team] / iterations,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("points", ascending=False).reset_index(drop=True)
    df["position"] = range(1, len(df) + 1)
    df["points"] = df["points"].round().astype(int)
    return df[["position", "team", "points", "title", "relegation"]]
