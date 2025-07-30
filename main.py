"""Command-line interface for running Brasileir\u00e3o simulations."""

# pylint: disable=wrong-import-position

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse

import numpy as np
import pandas as pd

from brasileirao import parse_matches, summary_table
from brasileirao.simulator import (
    DEFAULT_HOME_FIELD_ADVANTAGE,
    DEFAULT_JOBS,
    DEFAULT_ALPHA,
    DEFAULT_TIE_PERCENT,
)

# Default behaviour uses a simple model without recalculating parameters
DEFAULT_DYNAMIC_PARAMS = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Brasileirão 2025 title odds")
    parser.add_argument(
        "--file", default="data/Brasileirao2025A.txt", help="fixture file path"
    )
    parser.add_argument(
        "--simulations", type=int, default=5000, help="number of simulation runs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for repeatable simulations",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        default=True,
        help="disable the progress bar during simulations",
    )
    parser.add_argument(
        "--tie-percent",
        type=float,
        default=DEFAULT_TIE_PERCENT,
        help="percent of games ending in a tie",
    )
    parser.add_argument(
        "--home-advantage",
        type=float,
        default=DEFAULT_HOME_FIELD_ADVANTAGE,
        help="home field advantage factor",
    )
    parser.add_argument(
        "--dynamic-params",
        dest="dynamic_params",
        action="store_true",
        default=DEFAULT_DYNAMIC_PARAMS,
        help="estimate tie percent and home advantage from played matches",
    )
    def alpha_type(value: str) -> float:
        try:
            val = float(value)
        except ValueError as exc:  # pragma: no cover - argparse handles error
            raise argparse.ArgumentTypeError("alpha must be a number") from exc
        if not 0.0 <= val <= 1.0:
            raise argparse.ArgumentTypeError(
                "alpha must be between 0 and 1"
            )
        return val
    parser.add_argument(
        "--alpha",
        type=alpha_type,
        default=DEFAULT_ALPHA,
        help="smoothing factor for dynamic team parameters",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help="number of parallel workers",
    )
    parser.add_argument(
        "--html-output",
        default=os.path.join(os.path.dirname(__file__), "brasileirao.html"),
        help="path to save summary table as HTML",
    )
    args = parser.parse_args()

    matches = parse_matches(args.file)
    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    if args.dynamic_params:
        played = matches.dropna(subset=["home_score", "away_score"])
        if len(played) == 0:
            tie_prob = DEFAULT_TIE_PERCENT / 100.0
            home_adv = DEFAULT_HOME_FIELD_ADVANTAGE
        else:
            draws = (played["home_score"] == played["away_score"]).sum()
            tie_prob = draws / len(played)
            home_wins = (played["home_score"] > played["away_score"]).sum()
            away_wins = (played["home_score"] < played["away_score"]).sum()
            if away_wins == 0:
                home_adv = float(home_wins) if home_wins > 0 else 1.0
            else:
                home_adv = home_wins / away_wins
    else:
        tie_prob = args.tie_percent / 100.0
        home_adv = args.home_advantage

    tie_map = None
    home_map = None
    summary = summary_table(
        matches,
        iterations=args.simulations,
        rng=rng,
        progress=args.progress,
        tie_prob=tie_prob,
        home_field_adv=home_adv,
        tie_prob_map=tie_map,
        home_advantage_map=home_map,
        alpha=args.alpha,
        n_jobs=args.jobs,
    )
    if args.html_output:
        summary.to_html(args.html_output, index=False)

    TITLE_W = 7
    REL_W = 10
    POINTS_W = len("Pontos Esperados")
    print(
        f"{'Pos':>3}  {'Team':15s} {'Pontos Esperados':^{POINTS_W}} {'Title':^{TITLE_W}} {'Relegation':^{REL_W}}"
    )
    for _, row in summary.iterrows():
        title = f"{row['title']:.2%}"
        releg = f"{row['relegation']:.2%}"
        print(
            f"{row['position']:>2d}   {row['team']:15s} {row['points']:^{POINTS_W}d} {title:^{TITLE_W}} {releg:^{REL_W}}"
        )


if __name__ == "__main__":
    main()
