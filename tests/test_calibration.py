import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from calibration import estimate_parameters


def test_estimate_parameters_repeatable():
    tie, ha = estimate_parameters(["data/Brasileirao2024A.txt"])
    assert round(tie, 4) == 26.5789
    assert round(ha, 4) == 1.8182

