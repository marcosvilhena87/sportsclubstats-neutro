"""Automatic parameter estimation utilities."""

from __future__ import annotations

from typing import List

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
