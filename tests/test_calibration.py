import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from calibration import (
    estimate_parameters,
    estimate_team_strengths,
    estimate_goal_means,
    estimate_rho,
)


def test_estimate_parameters_repeatable():
    tie, ha = estimate_parameters(["data/Brasileirao2024A.txt"])
    assert round(tie, 4) == 26.5789
    assert round(ha, 4) == 1.8182


def test_estimate_parameters_multiple_files_repeatable():
    tie, ha = estimate_parameters([
        "data/Brasileirao2023A.txt",
        "data/Brasileirao2024A.txt",
    ])
    assert round(tie, 4) == 26.1842
    assert round(ha, 4) == 1.7635


def test_estimate_parameters_decay_zero_matches_latest_only():
    tie_latest, ha_latest = estimate_parameters(["data/Brasileirao2024A.txt"])
    tie_decay, ha_decay = estimate_parameters(
        ["data/Brasileirao2024A.txt", "data/Brasileirao2023A.txt"], decay=0.0
    )
    assert tie_latest == tie_decay
    assert ha_latest == ha_decay


def test_estimate_team_strengths_repeatable():
    strengths = estimate_team_strengths(["data/Brasileirao2024A.txt"])
    assert len(strengths) == 20
    assert round(strengths["Palmeiras"][0], 4) == 1.2917
    assert round(strengths["Fluminense"][1], 4) == 0.8396


def test_estimate_team_strengths_decay_zero_matches_latest_only():
    s_latest = estimate_team_strengths(["data/Brasileirao2024A.txt"])
    s_decay = estimate_team_strengths(
        ["data/Brasileirao2024A.txt", "data/Brasileirao2023A.txt"], decay=0.0
    )
    assert s_latest == s_decay


def test_estimate_goal_means_repeatable():
    hm, am = estimate_goal_means(["data/Brasileirao2024A.txt"])
    assert round(hm, 4) == 1.4105
    assert round(am, 4) == 1.0342


def test_estimate_goal_means_multiple_files_repeatable():
    hm, am = estimate_goal_means([
        "data/Brasileirao2023A.txt",
        "data/Brasileirao2024A.txt",
    ])
    assert round(hm, 4) == 1.4145
    assert round(am, 4) == 1.0526


def test_estimate_goal_means_decay_zero_matches_latest_only():
    hm_latest, am_latest = estimate_goal_means(["data/Brasileirao2024A.txt"])
    hm_decay, am_decay = estimate_goal_means(
        ["data/Brasileirao2024A.txt", "data/Brasileirao2023A.txt"], decay=0.0
    )
    assert hm_latest == hm_decay
    assert am_latest == am_decay


def test_estimate_rho_repeatable():
    rho = estimate_rho(["data/Brasileirao2024A.txt"])
    assert round(rho, 4) == -0.06


def test_estimate_rho_multiple_files_repeatable():
    rho = estimate_rho([
        "data/Brasileirao2023A.txt",
        "data/Brasileirao2024A.txt",
    ])
    assert round(rho, 4) == -0.014


def test_estimate_rho_decay_zero_matches_latest_only():
    rho_latest = estimate_rho(["data/Brasileirao2024A.txt"])
    rho_decay = estimate_rho(
        ["data/Brasileirao2024A.txt", "data/Brasileirao2023A.txt"], decay=0.0
    )
    assert rho_latest == rho_decay

