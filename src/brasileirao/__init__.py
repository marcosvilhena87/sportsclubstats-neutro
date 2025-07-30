"""Convenience exports for the simulator package."""

from .simulator import (
    league_table,
    parse_matches,
    reset_results_from,
    simulate_chances,
    simulate_relegation_chances,
    simulate_final_table,
    summary_table,
)

__all__ = [
    "parse_matches",
    "league_table",
    "reset_results_from",
    "simulate_chances",
    "simulate_relegation_chances",
    "simulate_final_table",
    "summary_table",
]
