"""Semiconductor supply chains, traced bottom up from an end market.

Each chain starts at the companies that sell to the end market (level 0) and
goes back through the chips they buy, the fabs and packaging houses that make
those chips, and the tools, software and materials the fabs need. Only
US-listed companies are included (ADRs such as TSM, ASML and SONY count, and so do the
OTC ADRs ASMVY and BESIY).

Links are curated by hand from public reporting (annual-report customer
disclosures, teardowns, supply agreements). They show the main relationships,
not every one. An edge points from supplier to customer.
"""

import numpy as np
import pandas as pd

from analysis.screener import index_trend, stock_snapshot

LEVELS = ["End market & systems", "Chips & components", "Memory, fabs & packaging", "Tools, software & materials"]
LEVEL_SHORT_NAMES = ["End market", "Chips", "Fabs & memory", "Tools & IP"]
ALL_CHAINS = "All end markets"

COMPANIES = {
    "AAPL": ("Apple", "iPhone maker; designs its own A-series processors, C1 modem and N1 wireless chip."),
    "GOOGL": ("Alphabet", "Pixel phones and Google Cloud; designs its own Tensor phone chip, TPU AI chips and Axion CPU."),
    "MSFT": ("Microsoft", "Azure cloud; one of the largest buyers of AI accelerators. Designs its own Maia and Cobalt chips."),
    "AMZN": ("Amazon", "AWS cloud; designs its own Trainium AI chips and Graviton CPUs."),
    "META": ("Meta Platforms", "AI data centers for its apps and models; designs its own MTIA accelerators."),
    "ORCL": ("Oracle", "Oracle Cloud; fast-growing buyer of AI accelerators."),
    "DELL": ("Dell Technologies", "AI servers built on NVIDIA and AMD accelerators."),
    "SMCI": ("Super Micro Computer", "AI servers and racks built on NVIDIA accelerators."),
    "ANET": ("Arista Networks", "Data center Ethernet switches; Microsoft and Meta are its largest customers."),
    "TSLA": ("Tesla", "EVs; designs its own self-driving computer chips and trains on NVIDIA GPUs."),
    "GM": ("General Motors", "Car maker; working with NVIDIA on driver assistance and self-driving."),
    "F": ("Ford Motor", "Car maker; BlueCruise driver assistance."),
    "RIVN": ("Rivian Automotive", "EV maker; Gen 2 vehicles run on NVIDIA Drive."),
    "NVDA": ("NVIDIA", "AI data center GPUs and systems; Drive computers for cars."),
    "AMD": ("Advanced Micro Devices", "Instinct AI accelerators and EPYC server CPUs."),
    "MRVL": ("Marvell Technology", "Custom AI chips for cloud providers and optical interconnect chips."),
    "INTC": ("Intel", "Xeon server CPUs; also runs its own fabs."),
    "ALAB": ("Astera Labs", "PCIe and CXL connectivity chips that link GPUs, CPUs and memory in AI servers."),
    "CRDO": ("Credo Technology", "Active electrical cables and connectivity chips for AI clusters."),
    "MBLY": ("Mobileye Global", "EyeQ chips for driver assistance, sold mostly through tier-1 suppliers."),
    "NXPI": ("NXP Semiconductors", "Automotive processors, microcontrollers, radar and networking chips."),
    "ADI": ("Analog Devices", "Analog and battery management chips, including GM's wireless battery management system."),
    "STM": ("STMicroelectronics", "Silicon carbide power chips and microcontrollers; a key Tesla supplier."),
    "ON": ("onsemi", "Silicon carbide power chips and image sensors for cars."),
    "QCOM": ("Qualcomm", "Modems and Snapdragon processors; iPhone modem supplier while Apple moves to its own."),
    "AVGO": ("Broadcom", "RF filters (FBAR) and wireless chips."),
    "SWKS": ("Skyworks Solutions", "RF front-end modules; Apple is its largest customer. Took over Qorvo in Oct 2026."),
    "CRUS": ("Cirrus Logic", "Audio codecs, amplifiers and haptic drivers; most revenue comes from Apple."),
    "TXN": ("Texas Instruments", "Power management and analog chips; runs its own fabs."),
    "MU": ("Micron Technology", "DRAM, high-bandwidth memory (HBM) for AI accelerators, and NAND flash; runs its own fabs."),
    "SONY": ("Sony Group (ADR)", "Camera image sensors; the main iPhone camera sensor supplier."),
    "TSM": ("TSMC (ADR)", "Leading-edge foundry and CoWoS advanced packaging; makes most of the world's AI accelerators and phone processors."),
    "GFS": ("GlobalFoundries", "Foundry for RF, mixed-signal and mature-node chips."),
    "AMKR": ("Amkor Technology", "Chip packaging and test; Apple and Qualcomm are its largest customers."),
    "ASX": ("ASE Technology (ADR)", "Packaging, test and system-in-package modules."),
    "ASML": ("ASML (ADR)", "Lithography; sole maker of EUV scanners for leading-edge nodes."),
    "AMAT": ("Applied Materials", "Deposition, etch and other wafer fab equipment."),
    "LRCX": ("Lam Research", "Etch and deposition equipment; large memory exposure."),
    "KLAC": ("KLA", "Inspection and process control equipment."),
    "TER": ("Teradyne", "Automated test equipment for processors and RF chips."),
    "ENTG": ("Entegris", "Specialty materials, filtration and wafer handling for fabs."),
    "ASMVY": ("ASMPT (OTC ADR)", "Back-end equipment: thermocompression and hybrid bonders, die and wire bonders."),
    "BESIY": ("BE Semiconductor (OTC ADR)", "Back-end equipment: hybrid bonders for 3D chip stacking and die attach."),
    "KLIC": ("Kulicke & Soffa", "Back-end equipment: wire bonders and thermocompression bonders for advanced packaging."),
    "SNPS": ("Synopsys", "Chip design software (EDA) and design IP."),
    "CDNS": ("Cadence Design Systems", "Chip design software (EDA) and design IP."),
    "ARM": ("Arm Holdings", "CPU architecture and core designs licensed to chip designers."),
}

CHAINS = {
    "Mobile devices": {
        "levels": [
            ["AAPL", "GOOGL"],
            ["QCOM", "AVGO", "SWKS", "CRUS", "TXN", "SONY"],
            ["TSM", "MU", "GFS", "AMKR", "ASX"],
            ["ARM", "SNPS", "CDNS", "ASML", "AMAT", "LRCX", "KLAC", "TER", "ASMVY", "BESIY", "KLIC", "ENTG"],
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
            ("ASMVY", "TSM", "Bonders for advanced packaging"),
            ("ASMVY", "AMKR", "Die bonders, wire bonders and thermocompression bonders"),
            ("ASMVY", "ASX", "Die bonders, wire bonders and thermocompression bonders"),
            ("BESIY", "TSM", "Hybrid bonders for 3D chip stacking (SoIC)"),
            ("BESIY", "AMKR", "Die attach equipment"),
            ("BESIY", "ASX", "Die attach equipment"),
            ("KLIC", "TSM", "Thermocompression bonders for advanced packaging"),
            ("KLIC", "AMKR", "Wire bonders and advanced packaging bonders"),
            ("KLIC", "ASX", "Wire bonders and advanced packaging bonders"),
            ("ENTG", "TSM", "Specialty materials and filtration"),
            ("ENTG", "MU", "Specialty materials and filtration"),
        ],
    },
    "AI / data center": {
        "levels": [
            ["MSFT", "AMZN", "GOOGL", "META", "ORCL", "DELL", "SMCI", "ANET"],
            ["NVDA", "AMD", "AVGO", "MRVL", "INTC", "ALAB", "CRDO"],
            ["TSM", "MU", "AMKR"],
            ["ARM", "SNPS", "CDNS", "ASML", "AMAT", "LRCX", "KLAC", "ASMVY", "BESIY", "KLIC", "ENTG"],
        ],
        "edges": [
            ("NVDA", "MSFT", "GPUs for Azure"),
            ("NVDA", "AMZN", "GPUs for AWS"),
            ("NVDA", "GOOGL", "GPUs for Google Cloud"),
            ("NVDA", "META", "GPUs for its AI clusters"),
            ("NVDA", "ORCL", "GPUs for Oracle Cloud"),
            ("NVDA", "DELL", "GPUs and HGX boards for AI servers"),
            ("NVDA", "SMCI", "GPUs and HGX boards for AI servers"),
            ("AMD", "MSFT", "Instinct accelerators for Azure"),
            ("AMD", "META", "Instinct accelerators"),
            ("AMD", "ORCL", "Instinct accelerators for Oracle Cloud"),
            ("AMD", "DELL", "EPYC CPUs and Instinct accelerators"),
            ("AVGO", "GOOGL", "Co-designs Google's TPU AI chips"),
            ("AVGO", "META", "Custom AI chips and networking"),
            ("AVGO", "ANET", "Tomahawk and Jericho switch chips"),
            ("MRVL", "AMZN", "Custom AI chips and optical interconnect chips"),
            ("MRVL", "MSFT", "Custom AI accelerator work"),
            ("INTC", "DELL", "Xeon server CPUs"),
            ("INTC", "SMCI", "Xeon server CPUs"),
            ("ALAB", "AMZN", "PCIe and CXL connectivity chips"),
            ("CRDO", "AMZN", "Active electrical cables"),
            ("CRDO", "MSFT", "Active electrical cables"),
            ("TSM", "NVDA", "Makes the GPUs and packages them with CoWoS"),
            ("TSM", "AMD", "Makes Instinct and EPYC chips"),
            ("TSM", "AVGO", "Makes custom AI and switch chips"),
            ("TSM", "MRVL", "Makes Marvell's custom chips"),
            ("MU", "NVDA", "HBM memory stacked next to the GPU"),
            ("MU", "AMD", "HBM memory for Instinct accelerators"),
            ("AMKR", "NVDA", "Advanced packaging and test"),
            ("ARM", "NVDA", "Arm cores in the Grace CPU"),
            ("ARM", "AMZN", "Arm Neoverse cores in Graviton"),
            ("ARM", "MSFT", "Arm Neoverse cores in Cobalt"),
            ("ARM", "GOOGL", "Arm Neoverse cores in Axion"),
            ("SNPS", "NVDA", "Design software and IP"),
            ("SNPS", "AMD", "Design software and IP"),
            ("CDNS", "NVDA", "Design software and IP"),
            ("CDNS", "AMD", "Design software and IP"),
            ("ASML", "TSM", "EUV and DUV lithography"),
            ("ASML", "MU", "EUV and DUV lithography for DRAM"),
            ("ASML", "INTC", "EUV lithography, including High-NA"),
            ("AMAT", "TSM", "Wafer fab equipment"),
            ("AMAT", "MU", "Wafer fab equipment"),
            ("AMAT", "INTC", "Wafer fab equipment"),
            ("LRCX", "TSM", "Etch and deposition"),
            ("LRCX", "MU", "Etch and deposition for DRAM and HBM"),
            ("KLAC", "TSM", "Inspection and process control"),
            ("KLAC", "MU", "Inspection and process control"),
            ("ASMVY", "TSM", "Thermocompression and hybrid bonders for advanced packaging"),
            ("ASMVY", "AMKR", "Bonders for advanced packaging"),
            ("BESIY", "TSM", "Hybrid bonders for 3D chip stacking (SoIC)"),
            ("KLIC", "TSM", "Thermocompression bonders for advanced packaging"),
            ("KLIC", "AMKR", "Advanced packaging bonders"),
            ("ENTG", "TSM", "Specialty materials and filtration"),
            ("ENTG", "MU", "Specialty materials and filtration"),
        ],
    },
    "Automotive": {
        "levels": [
            ["TSLA", "GM", "F", "RIVN"],
            ["NVDA", "MBLY", "QCOM", "NXPI", "ADI", "STM", "ON"],
            ["TSM", "GFS"],
            ["ARM", "SNPS", "CDNS", "ASML", "AMAT", "LRCX", "KLAC", "ENTG"],
        ],
        "edges": [
            ("NVDA", "TSLA", "GPUs for training its self-driving models"),
            ("NVDA", "GM", "Drive computers and AI training for self-driving"),
            ("NVDA", "RIVN", "Drive Orin computers in Gen 2 vehicles"),
            ("MBLY", "F", "EyeQ driver-assistance chips"),
            ("QCOM", "GM", "Snapdragon cockpit chips"),
            ("NXPI", "F", "Processors for the SYNC infotainment system"),
            ("ADI", "GM", "Wireless battery management system"),
            ("STM", "TSLA", "Silicon carbide power chips for the inverter"),
            ("ON", "TSLA", "Silicon carbide power chips"),
            ("TSM", "TSLA", "Makes Tesla's AI5 self-driving chip"),
            ("TSM", "NVDA", "Makes the Drive Orin and Thor chips"),
            ("TSM", "MBLY", "Makes EyeQ chips"),
            ("TSM", "QCOM", "Makes Snapdragon automotive chips"),
            ("TSM", "NXPI", "Makes NXP's advanced-node processors"),
            ("GFS", "GM", "Long-term supply agreement for GM's chips"),
            ("ARM", "NVDA", "Arm cores in Drive Thor"),
            ("ARM", "NXPI", "Arm cores in S32 processors"),
            ("SNPS", "NVDA", "Design software and IP"),
            ("CDNS", "NVDA", "Design software and IP"),
            ("ASML", "TSM", "EUV and DUV lithography"),
            ("AMAT", "TSM", "Wafer fab equipment"),
            ("AMAT", "GFS", "Wafer fab equipment"),
            ("LRCX", "TSM", "Etch and deposition"),
            ("KLAC", "TSM", "Inspection and process control"),
            ("ENTG", "TSM", "Specialty materials and filtration"),
        ],
    },
}


def _merge_chains(chains):
    """One graph of every chain: companies keep their level, a link in several chains keeps every note."""

    levels = [[] for _ in LEVELS]
    notes = {}
    for chain in chains.values():
        for index, level in enumerate(chain["levels"]):
            levels[index] += [ticker for ticker in level if ticker not in levels[index]]
        for source, target, note in chain["edges"]:
            notes.setdefault((source, target), [])
            if note not in notes[(source, target)]:
                notes[(source, target)].append(note)
    return {"levels": levels, "edges": [(source, target, "; ".join(parts)) for (source, target), parts in notes.items()]}


CHAINS[ALL_CHAINS] = _merge_chains(CHAINS)

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
