import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pandas as pd
import numpy as np
import pytest
from simulator import parse_matches, league_table, simulate_chances
from calibration import estimate_team_strengths
import simulator


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
    assert {
        "position",
        "team",
        "points",
        "wins",
        "gd",
        "title",
        "relegation",
    }.issubset(table1.columns)


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
    )
    assert table["draws"].sum() == 0


def _minimal_matches():
    played = pd.DataFrame(
        [],
        columns=["date", "home_team", "away_team", "home_score", "away_score"],
    )
    remaining = pd.DataFrame(
        [{"date": "2025-01-01", "home_team": "A", "away_team": "B"}]
    )
    return played, remaining


def test_simulate_table_invalid_tie_prob():
    played, remaining = _minimal_matches()
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        simulator._simulate_table(played, remaining, rng, tie_prob=-0.1)
    with pytest.raises(ValueError):
        simulator._simulate_table(played, remaining, rng, tie_prob=1.1)


def test_simulate_table_invalid_home_advantage():
    played, remaining = _minimal_matches()
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        simulator._simulate_table(played, remaining, rng, home_advantage=0)
    with pytest.raises(ValueError):
        simulator._simulate_table(played, remaining, rng, home_advantage=-1)


def test_simulate_chances_invalid_params():
    df = parse_matches("data/Brasileirao2024A.txt")
    with pytest.raises(ValueError):
        simulator.simulate_chances(df, iterations=1, tie_prob=2.0, progress=False)
    with pytest.raises(ValueError):
        simulator.simulate_chances(df, iterations=1, home_advantage=0, progress=False)


def test_simulate_final_table_custom_params_deterministic():
    df = parse_matches("data/Brasileirao2024A.txt")
    rng = np.random.default_rng(9)
    t1 = simulator.simulate_final_table(
        df,
        iterations=5,
        rng=rng,
        tie_prob=0.2,
        home_advantage=1.3,
        n_jobs=2,
    )
    rng = np.random.default_rng(9)
    t2 = simulator.simulate_final_table(
        df,
        iterations=5,
        rng=rng,
        tie_prob=0.2,
        home_advantage=1.3,
        n_jobs=2,
    )
    pd.testing.assert_frame_equal(t1, t2)


def test_custom_params_repeatable():
    df = parse_matches("data/Brasileirao2024A.txt")
    rng = np.random.default_rng(11)
    t1 = simulator.summary_table(
        df,
        iterations=5,
        rng=rng,
        tie_prob=0.25,
        home_advantage=1.4,
        progress=False,
        n_jobs=2,
    )
    rng = np.random.default_rng(11)
    t2 = simulator.summary_table(
        df,
        iterations=5,
        rng=rng,
        tie_prob=0.25,
        home_advantage=1.4,
        progress=False,
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


def test_reset_results_from():
    df = parse_matches("data/Brasileirao2024A.txt")
    start = "2024-07-01"
    reset = simulator.reset_results_from(df, start)
    mask = reset["date"] >= pd.to_datetime(start)
    assert reset.loc[mask, ["home_score", "away_score"]].isna().all().all()
    # ensure simulation runs without errors
    simulator.simulate_chances(reset, iterations=1, progress=False)


def test_summary_table_after_reset_deterministic():
    df = parse_matches("data/Brasileirao2024A.txt")
    df = simulator.reset_results_from(df, "2024-07-01")
    rng = np.random.default_rng(7)
    t1 = simulator.summary_table(
        df,
        iterations=5,
        rng=rng,
        tie_prob=0.22,
        home_advantage=1.2,
        progress=False,
        n_jobs=2,
    )
    rng = np.random.default_rng(7)
    t2 = simulator.summary_table(
        df,
        iterations=5,
        rng=rng,
        tie_prob=0.22,
        home_advantage=1.2,
        progress=False,
        n_jobs=2,
    )
    pd.testing.assert_frame_equal(t1, t2)


def test_summary_table_after_reset_other_params_deterministic():
    df = parse_matches("data/Brasileirao2024A.txt")
    df = simulator.reset_results_from(df, "2024-07-01")
    rng = np.random.default_rng(8)
    t1 = simulator.summary_table(
        df,
        iterations=5,
        rng=rng,
        tie_prob=0.3,
        home_advantage=1.5,
        progress=False,
        n_jobs=2,
    )
    rng = np.random.default_rng(8)
    t2 = simulator.summary_table(
        df,
        iterations=5,
        rng=rng,
        tie_prob=0.3,
        home_advantage=1.5,
        progress=False,
        n_jobs=2,
    )
    pd.testing.assert_frame_equal(t1, t2)


def test_team_params_repeatable():
    df = parse_matches("data/Brasileirao2024A.txt")
    params = estimate_team_strengths(["data/Brasileirao2024A.txt"])
    rng = np.random.default_rng(100)
    t1 = simulator.summary_table(
        df,
        iterations=5,
        rng=rng,
        team_params=params,
        progress=False,
        n_jobs=2,
    )
    rng = np.random.default_rng(100)
    t2 = simulator.summary_table(
        df,
        iterations=5,
        rng=rng,
        team_params=params,
        progress=False,
        n_jobs=2,
    )
    pd.testing.assert_frame_equal(t1, t2)


def test_poisson_mode_repeatable():
    df = parse_matches("data/Brasileirao2024A.txt")
    rng = np.random.default_rng(555)
    t1 = simulator.summary_table(
        df,
        iterations=5,
        rng=rng,
        home_goals_mean=1.5,
        away_goals_mean=1.2,
        progress=False,
        n_jobs=2,
    )
    rng = np.random.default_rng(555)
    t2 = simulator.summary_table(
        df,
        iterations=5,
        rng=rng,
        home_goals_mean=1.5,
        away_goals_mean=1.2,
        progress=False,
        n_jobs=2,
    )
    pd.testing.assert_frame_equal(t1, t2)


def test_simulate_table_invalid_goal_means():
    played, remaining = _minimal_matches()
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        simulator._simulate_table(played, remaining, rng, home_goals_mean=0)
    with pytest.raises(ValueError):
        simulator._simulate_table(played, remaining, rng, away_goals_mean=-1)



