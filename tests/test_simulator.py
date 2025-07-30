import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pandas as pd
import numpy as np
import pytest
from brasileirao import parse_matches, league_table, simulate_chances
from brasileirao import simulator


def test_parse_matches():
    df = parse_matches('data/Brasileirao2024A.txt')
    assert len(df) == 380
    assert {'home_team', 'away_team', 'home_score', 'away_score'}.issubset(df.columns)


def test_league_table():
    df = parse_matches('data/Brasileirao2024A.txt')
    table = league_table(df)
    assert 'points' in table.columns
    assert table['played'].max() > 0


def test_league_table_deterministic_sorting():
    data = [
        {'date': '2025-01-01', 'home_team': 'Alpha', 'away_team': 'Beta', 'home_score': 1, 'away_score': 0},
        {'date': '2025-01-02', 'home_team': 'Beta', 'away_team': 'Gamma', 'home_score': 1, 'away_score': 0},
        {'date': '2025-01-03', 'home_team': 'Gamma', 'away_team': 'Alpha', 'home_score': 1, 'away_score': 0},
    ]
    df = pd.DataFrame(data)
    table = league_table(df)
    assert list(table.team) == sorted(table.team)


def test_simulate_chances_sum_to_one():
    df = parse_matches('data/Brasileirao2024A.txt')
    chances = simulate_chances(df, iterations=10, progress=False)
    assert abs(sum(chances.values()) - 1.0) < 1e-6


def test_simulate_chances_seed_repeatability():
    df = parse_matches('data/Brasileirao2024A.txt')
    rng = np.random.default_rng(1234)
    chances1 = simulate_chances(
        df, iterations=5, rng=rng, progress=False, n_jobs=2
    )
    rng = np.random.default_rng(1234)
    chances2 = simulate_chances(
        df, iterations=5, rng=rng, progress=False, n_jobs=2
    )
    assert chances1 == chances2




def test_simulate_relegation_chances_sum_to_four():
    df = parse_matches('data/Brasileirao2024A.txt')
    probs = simulator.simulate_relegation_chances(df, iterations=10, progress=False)
    assert abs(sum(probs.values()) - 4.0) < 1e-6


def test_simulate_relegation_chances_seed_repeatability():
    df = parse_matches('data/Brasileirao2024A.txt')
    rng = np.random.default_rng(123)
    first = simulator.simulate_relegation_chances(
        df, iterations=5, rng=rng, progress=False, n_jobs=2
    )
    rng = np.random.default_rng(123)
    second = simulator.simulate_relegation_chances(
        df, iterations=5, rng=rng, progress=False, n_jobs=2
    )
    assert first == second


def test_simulate_final_table_deterministic():
    df = parse_matches('data/Brasileirao2024A.txt')
    rng = np.random.default_rng(1)
    table1 = simulator.simulate_final_table(
        df, iterations=5, rng=rng, progress=False, n_jobs=2
    )
    rng = np.random.default_rng(1)
    table2 = simulator.simulate_final_table(
        df, iterations=5, rng=rng, progress=False, n_jobs=2
    )
    pd.testing.assert_frame_equal(table1, table2)
    assert {"team", "position", "points"}.issubset(table1.columns)


def test_summary_table_deterministic():
    df = parse_matches('data/Brasileirao2024A.txt')
    rng = np.random.default_rng(5)
    table1 = simulator.summary_table(
        df, iterations=5, rng=rng, progress=False, n_jobs=2
    )
    rng = np.random.default_rng(5)
    table2 = simulator.summary_table(
        df, iterations=5, rng=rng, progress=False, n_jobs=2
    )
    pd.testing.assert_frame_equal(table1, table2)
    assert {"position", "team", "points", "title", "relegation"}.issubset(table1.columns)


def test_league_table_tiebreakers():
    data = [
        {"date": "2025-01-01", "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 2},
        {"date": "2025-01-02", "home_team": "A", "away_team": "C", "home_score": 1, "away_score": 0},
        {"date": "2025-01-03", "home_team": "C", "away_team": "A", "home_score": 0, "away_score": 1},
        {"date": "2025-01-04", "home_team": "B", "away_team": "C", "home_score": 3, "away_score": 0},
    ]
    df = pd.DataFrame(data)
    table = league_table(df)
    assert list(table.team[:2]) == ["B", "A"]


def test_simulate_table_no_draws_when_zero_tie():
    played = pd.DataFrame(
        [],
        columns=["date", "home_team", "away_team", "home_score", "away_score"],
    )
    remaining = pd.DataFrame(
        [
            {"date": "2025-01-01", "home_team": "A", "away_team": "B"},
            {"date": "2025-01-02", "home_team": "B", "away_team": "A"},
        ]
    )
    rng = np.random.default_rng(4)
    table = simulator._simulate_table(
        played,
        remaining,
        rng,
        tie_prob=0.0,
        home_field_adv=1.0,
    )
    assert table["draws"].sum() == 0


def test_simulate_final_table_custom_params_deterministic():
    df = parse_matches("data/Brasileirao2024A.txt")
    rng = np.random.default_rng(9)
    t1 = simulator.simulate_final_table(
        df,
        iterations=5,
        rng=rng,
        tie_prob=0.2,
        home_field_adv=1.5,
        n_jobs=2,
    )
    rng = np.random.default_rng(9)
    t2 = simulator.simulate_final_table(
        df,
        iterations=5,
        rng=rng,
        tie_prob=0.2,
        home_field_adv=1.5,
        n_jobs=2,
    )
    pd.testing.assert_frame_equal(t1, t2)


def test_progress_default_true(monkeypatch):
    df = parse_matches("data/Brasileirao2024A.txt")
    called = {}

    def fake_tqdm(iterable, **kwargs):
        called["used"] = True
        return iterable

    monkeypatch.setattr(simulator, "tqdm", fake_tqdm)
    simulator.simulate_chances(df, iterations=1)
    assert called.get("used", False)


def test_parallel_consistency():
    df = parse_matches("data/Brasileirao2024A.txt")
    rng = np.random.default_rng(6)
    serial = simulator.summary_table(
        df, iterations=5, rng=rng, progress=False, n_jobs=1
    )
    rng = np.random.default_rng(6)
    parallel = simulator.summary_table(
        df, iterations=5, rng=rng, progress=False, n_jobs=2
    )
    pd.testing.assert_frame_equal(serial, parallel)




def test_dynamic_params_deterministic():
    df = parse_matches("data/Brasileirao2024A.txt")
    rng = np.random.default_rng(42)
    t1 = simulator.summary_table(
        df,
        iterations=5,
        rng=rng,
        progress=False,
        n_jobs=2,
    )
    rng = np.random.default_rng(42)
    t2 = simulator.summary_table(
        df,
        iterations=5,
        rng=rng,
        progress=False,
        n_jobs=2,
    )
    pd.testing.assert_frame_equal(t1, t2)



def test_alpha_validation_private():
    played = pd.DataFrame(
        [],
        columns=["date", "home_team", "away_team", "home_score", "away_score"],
    )
    remaining = pd.DataFrame(
        [{"date": "2025-01-01", "home_team": "A", "away_team": "B"}]
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulator._simulate_table(played, remaining, rng, alpha=-0.1)
    with pytest.raises(ValueError):
        simulator._simulate_table(played, remaining, rng, alpha=1.1)
    simulator._simulate_table(played, remaining, rng, alpha=0.0)
    simulator._simulate_table(played, remaining, rng, alpha=1.0)


def test_alpha_validation_public():
    df = parse_matches("data/Brasileirao2024A.txt")
    with pytest.raises(ValueError):
        simulator.simulate_chances(df, iterations=1, progress=False, alpha=-0.2)
    with pytest.raises(ValueError):
        simulator.simulate_chances(df, iterations=1, progress=False, alpha=1.2)
    simulator.simulate_chances(df, iterations=1, progress=False, alpha=0.5)
