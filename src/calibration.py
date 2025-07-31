"""Automatic parameter estimation utilities."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from simulator import parse_matches


def estimate_parameters(paths: List[str]) -> tuple[float, float]:
    """Estimate draw rate and home advantage from past seasons.

    Parameters
    ----------
    paths:
        A list of text file paths containing fixture results in the
        SportsClubStats format.

    Returns
    -------
    tuple[float, float]
        The ``(tie_percent, home_advantage)`` calculated from the data.
    """

    total_games = 0
    home_wins = 0
    away_wins = 0
    draws = 0

    for path in paths:
        df = parse_matches(path)
        played = df.dropna(subset=["home_score", "away_score"])
        total_games += len(played)
        home_wins += (played["home_score"] > played["away_score"]).sum()
        away_wins += (played["home_score"] < played["away_score"]).sum()
        draws += (played["home_score"] == played["away_score"]).sum()

    if total_games == 0:
        raise ValueError("No played games found in provided paths")

    tie_percent = float(100.0 * draws / total_games)
    if away_wins == 0:
        home_advantage = 1.0
    else:
        home_advantage = float(home_wins / away_wins)

    return tie_percent, home_advantage


def estimate_team_strengths(paths: List[str]) -> Dict[str, tuple[float, float]]:
    """Estimate attack and defense multipliers for each team.

    The returned values are normalized so that ``1.0`` represents league
    average performance.  Values greater than 1.0 indicate stronger attack or
    weaker defense respectively.

    Parameters
    ----------
    paths:
        A list of text file paths containing historical fixtures with results.

    Returns
    -------
    Dict[str, tuple[float, float]]
        Mapping of team name to ``(attack, defense)`` multipliers.
    """

    df = [parse_matches(p) for p in paths]
    if not df:
        return {}

    played = (
        pd.concat(df, ignore_index=True)
        .dropna(subset=["home_score", "away_score"])
    )

    teams = pd.unique(played[["home_team", "away_team"]].values.ravel())
    stats = {t: {"gf": 0, "ga": 0, "games": 0} for t in teams}

    for _, row in played.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        stats[ht]["gf"] += hs
        stats[ht]["ga"] += as_
        stats[ht]["games"] += 1
        stats[at]["gf"] += as_
        stats[at]["ga"] += hs
        stats[at]["games"] += 1

    total_goals = sum(s["gf"] for s in stats.values())
    total_games = sum(s["games"] for s in stats.values())
    if total_games == 0:
        raise ValueError("No played games found in provided paths")

    avg_goals = total_goals / total_games

    strengths: Dict[str, tuple[float, float]] = {}
    for team, s in stats.items():
        attack = (s["gf"] / s["games"]) / avg_goals
        defense = (s["ga"] / s["games"]) / avg_goals
        strengths[team] = (attack, defense)

    return strengths
