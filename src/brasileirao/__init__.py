"""Convenience exports for the simulator package."""

from .simulator import (
    league_table,
    parse_matches,
    simulate_chances,
    simulate_relegation_chances,
    simulate_final_table,
    summary_table,
)

__all__ = [
    "parse_matches",
    "league_table",
    "simulate_chances",
    "simulate_relegation_chances",
    "simulate_final_table",
    "summary_table",
]
