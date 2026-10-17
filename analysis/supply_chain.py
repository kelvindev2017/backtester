"""Semiconductor supply chains, traced bottom up from an end market.

Each chain starts at the companies that sell to the end market (level 0) and
goes back through the chips they buy, the fabs and packaging houses that make
those chips, and the tools, software and materials the fabs need. Only
US-listed companies are included (ADRs such as TSM, ASML and SONY count).

Links are curated by hand from public reporting (annual-report customer
disclosures, teardowns, supply agreements). They show the main relationships,
not every one. An edge points from supplier to customer.
"""

import numpy as np
import pandas as pd

from analysis.screener import index_trend, stock_snapshot

LEVELS = ["End market", "Chips & components", "Fabs & packaging", "Tools, software & materials"]
LEVEL_SHORT_NAMES = ["End market", "Chips", "Fabs", "Tools & IP"]

COMPANIES = {
    "AAPL": ("Apple", "iPhone maker; designs its own A-series processors, C1 modem and N1 wireless chip."),
    "GOOGL": ("Alphabet", "Pixel phones; designs its own Tensor processor."),
    "QCOM": ("Qualcomm", "Modems and Snapdragon processors; iPhone modem supplier while Apple moves to its own."),
    "AVGO": ("Broadcom", "RF filters (FBAR) and wireless chips."),
    "SWKS": ("Skyworks Solutions", "RF front-end modules; Apple is its largest customer. Took over Qorvo in Oct 2026."),
    "CRUS": ("Cirrus Logic", "Audio codecs, amplifiers and haptic drivers; most revenue comes from Apple."),
    "TXN": ("Texas Instruments", "Power management and analog chips; runs its own fabs."),
    "MU": ("Micron Technology", "Mobile DRAM (LPDDR) and NAND flash; runs its own fabs."),
    "SONY": ("Sony Group (ADR)", "Camera image sensors; the main iPhone camera sensor supplier."),
    "TSM": ("TSMC (ADR)", "Foundry that makes Apple's A-series, Qualcomm's Snapdragon and Google's Tensor G5."),
    "GFS": ("GlobalFoundries", "Foundry for RF, mixed-signal and mature-node chips."),
    "AMKR": ("Amkor Technology", "Chip packaging and test; Apple and Qualcomm are its largest customers."),
    "ASX": ("ASE Technology (ADR)", "Packaging, test and system-in-package modules."),
    "ASML": ("ASML (ADR)", "Lithography; sole maker of EUV scanners for leading-edge nodes."),
    "AMAT": ("Applied Materials", "Deposition, etch and other wafer fab equipment."),
    "LRCX": ("Lam Research", "Etch and deposition equipment; large memory exposure."),
    "KLAC": ("KLA", "Inspection and process control equipment."),
    "TER": ("Teradyne", "Automated test equipment for processors and RF chips."),
    "ENTG": ("Entegris", "Specialty materials, filtration and wafer handling for fabs."),
    "SNPS": ("Synopsys", "Chip design software (EDA) and design IP."),
    "CDNS": ("Cadence Design Systems", "Chip design software (EDA) and design IP."),
    "ARM": ("Arm Holdings", "CPU architecture and core designs licensed to chip designers."),
}

CHAINS = {
    "Mobile devices": {
        "levels": [
            ["AAPL", "GOOGL"],
            ["QCOM", "AVGO", "SWKS", "CRUS", "TXN", "MU", "SONY"],
            ["TSM", "GFS", "AMKR", "ASX"],
            ["ARM", "SNPS", "CDNS", "ASML", "AMAT", "LRCX", "KLAC", "TER", "ENTG"],
        ],
        "edges": [
            ("QCOM", "AAPL", "Cellular modems (Apple is shifting to its own C1 modem)"),
            ("AVGO", "AAPL", "RF filters and wireless components"),
            ("SWKS", "AAPL", "RF front-end modules (including the former Qorvo business)"),
            ("CRUS", "AAPL", "Audio and haptic chips"),
            ("TXN", "AAPL", "Power management and analog chips"),
            ("MU", "AAPL", "Mobile DRAM and NAND"),
            ("SONY", "AAPL", "Camera image sensors"),
            ("TSM", "AAPL", "Makes the A-series processors and Apple's own modem and wireless chips"),
            ("TSM", "GOOGL", "Makes the Tensor G5 processor (from Pixel 10)"),
            ("TSM", "QCOM", "Makes Snapdragon processors and modems"),
            ("TSM", "AVGO", "Makes Broadcom's wireless chips"),
            ("GFS", "QCOM", "RF and mature-node chips under a long-term supply agreement"),
            ("GFS", "CRUS", "Makes Cirrus Logic's mixed-signal chips"),
            ("AMKR", "AAPL", "Packaging and test (Apple is its largest customer)"),
            ("AMKR", "QCOM", "Packaging and test"),
            ("ASX", "AAPL", "System-in-package modules"),
            ("ASX", "QCOM", "Packaging and test"),
            ("ARM", "AAPL", "Arm architecture license"),
            ("ARM", "QCOM", "Arm architecture license"),
            ("ARM", "GOOGL", "Arm CPU cores in Tensor"),
            ("SNPS", "AAPL", "Design software and IP"),
            ("SNPS", "QCOM", "Design software and IP"),
            ("CDNS", "AAPL", "Design software and IP"),
            ("CDNS", "QCOM", "Design software and IP"),
            ("ASML", "TSM", "EUV and DUV lithography"),
            ("ASML", "MU", "EUV and DUV lithography for DRAM"),
            ("AMAT", "TSM", "Wafer fab equipment"),
            ("AMAT", "MU", "Wafer fab equipment"),
            ("AMAT", "GFS", "Wafer fab equipment"),
            ("LRCX", "TSM", "Etch and deposition"),
            ("LRCX", "MU", "Etch and deposition for DRAM and NAND"),
            ("KLAC", "TSM", "Inspection and process control"),
            ("KLAC", "MU", "Inspection and process control"),
            ("TER", "TSM", "Test systems for mobile processors"),
            ("TER", "AMKR", "Test systems for packaged chips"),
            ("ENTG", "TSM", "Specialty materials and filtration"),
            ("ENTG", "MU", "Specialty materials and filtration"),
        ],
    },
}

NODE_METRICS = ["Return 1m %", "Return 3m %", "Return 6m %", "Return YTD %", "Return 1y %", "% from 52w high"]


def chain_tickers(chain):
    """Every ticker in a chain, level by level."""

    return [ticker for level in CHAINS[chain]["levels"] for ticker in level]


def node_positions(chain):
    """(x, y) per ticker: x is the level (0 = end market), y spreads each level evenly over [0, 1], top first."""

    positions = {}
    for x, level in enumerate(CHAINS[chain]["levels"]):
        count = len(level)
        for row, ticker in enumerate(level):
            positions[ticker] = (x, 0.5 if count == 1 else 1 - row / (count - 1))
    return positions


def neighbors(chain, ticker):
    """(suppliers, customers) of a ticker: lists of (ticker, note)."""

    edges = CHAINS[chain]["edges"]
    suppliers = [(source, note) for source, target, note in edges if target == ticker]
    customers = [(target, note) for source, target, note in edges if source == ticker]
    return suppliers, customers


def linked_tickers(chain, ticker):
    """The ticker plus everything upstream and downstream of it, following the edges both ways."""

    edges = CHAINS[chain]["edges"]
    linked = {ticker}
    for direction in (0, 1):
        frontier = [ticker]
        while frontier:
            current = frontier.pop()
            for edge in edges:
                if edge[1 - direction] == current and edge[direction] not in linked:
                    linked.add(edge[direction])
                    frontier.append(edge[direction])
    return linked


def node_table(chain, prices):
    """Price snapshot per company in the chain (one row per ticker, NaN where Yahoo returned nothing)."""

    rows = {}
    for ticker in chain_tickers(chain):
        frame = prices.get(ticker)
        row = stock_snapshot(frame) if frame is not None and not frame.empty else {}
        row["Trend"] = index_trend(frame) if row else "No data"
        rows[ticker] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    for column in ["Close"] + NODE_METRICS:
        if column not in table:
            table[column] = np.nan
    names = {ticker: COMPANIES[ticker][0] for ticker in table.index}
    table.insert(0, "Company", pd.Series(names))
    positions = node_positions(chain)
    table.insert(1, "Level", [LEVELS[positions[ticker][0]] for ticker in table.index])
    return table
