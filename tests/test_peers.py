import pandas as pd

from analysis.peers import merge_peers, rank_by_size, target_weight

SEMI_EQUIPMENT = pd.Series({"AMAT": 0.31, "LRCX": 0.30, "KLAC": 0.21, "TER": 0.043, "Q": 0.027, "ENTG": 0.024,
                            "NVMI": 0.011, "ONTO": 0.011})


def test_peers_closest_in_size_come_first():
    # ASMPT is about the size of Entegris and Qnity, far smaller than Applied Materials.
    assert rank_by_size(SEMI_EQUIPMENT, target=0.022, count=4) == ["ENTG", "Q", "TER", "NVMI"]
    assert rank_by_size(SEMI_EQUIPMENT, count=3) == ["AMAT", "LRCX", "KLAC"]
    assert rank_by_size(SEMI_EQUIPMENT, target=0.30, count=2, exclude=["lrcx"]) == ["AMAT", "KLAC"]
    assert rank_by_size(pd.Series(dtype=float), target=1.0) == []


def test_target_weight_from_own_row_or_market_cap():
    assert target_weight(SEMI_EQUIPMENT, "KLAC") == 0.21
    # A US$5.4B company against Applied Materials' US$310B at 0.31 weight.
    assert target_weight(SEMI_EQUIPMENT, "ASMVY", 5.4e9, ("AMAT", 310e9)) == 0.0054
    assert target_weight(SEMI_EQUIPMENT, "ASMVY") is None


def test_merge_keeps_the_best_source_first_without_duplicates_or_the_stock():
    assert merge_peers([["V", "MA", "GPN"], ["ma", "PYPL", "FIS", "FI"]], "GPN", count=4) == ["V", "MA", "PYPL", "FIS"]
