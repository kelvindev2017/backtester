import numpy as np

from analysis.supply_chain import (
    CHAINS,
    COMPANIES,
    LEVELS,
    chain_tickers,
    linked_tickers,
    neighbors,
    node_positions,
    node_table,
)
from tests.conftest import make_bars


def test_chains_are_consistent():
    for chain in CHAINS.values():
        assert len(chain["levels"]) == len(LEVELS)
        tickers = [ticker for level in chain["levels"] for ticker in level]
        assert len(tickers) == len(set(tickers)), "a company appears on two levels"
        assert set(tickers) <= set(COMPANIES)
        level_of = {ticker: index for index, level in enumerate(chain["levels"]) for ticker in level}
        pairs = set()
        for source, target, note in chain["edges"]:
            assert source in level_of and target in level_of, (source, target)
            # Edges run from supplier to customer, so the supplier sits further from the end market.
            assert level_of[source] > level_of[target], (source, target)
            assert note
            assert (source, target) not in pairs
            pairs.add((source, target))
        linked = {ticker for edge in chain["edges"] for ticker in edge[:2]}
        assert linked == set(tickers), "every company has at least one link"


def test_positions_spread_each_level():
    positions = node_positions("Mobile devices")
    assert positions["AAPL"] == (0, 1.0)
    assert positions["GOOGL"] == (0, 0.0)
    assert {x for x, _ in positions.values()} == {0, 1, 2, 3}
    assert all(0 <= y <= 1 for _, y in positions.values())


def test_neighbors_and_linked_chain():
    suppliers, customers = neighbors("Mobile devices", "TSM")
    assert "ASML" in [ticker for ticker, _ in suppliers]
    assert "AAPL" in [ticker for ticker, _ in customers]
    linked = linked_tickers("Mobile devices", "GOOGL")
    # Upstream through TSM and ARM to the toolmakers, but not Apple's other suppliers.
    assert {"GOOGL", "TSM", "ARM", "ASML"} <= linked
    assert "CRUS" not in linked and "AAPL" not in linked


def test_node_table_handles_missing_prices():
    tickers = chain_tickers("Mobile devices")
    prices = {ticker: make_bars(np.linspace(100, 130, 300), seed=1) for ticker in tickers[:-1]}
    table = node_table("Mobile devices", prices)
    assert list(table.index) == tickers
    assert table.loc[tickers[0], "Return 3m %"] > 0
    assert np.isnan(table.loc[tickers[-1], "Return 3m %"])
    assert table.loc[tickers[-1], "Trend"] == "No data"
