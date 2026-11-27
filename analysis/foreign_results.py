"""Quarterly results of foreign filers (20-F companies such as ASML and TSMC) read from their 6-K results releases.

SEC Company Facts holds only their yearly figures, so the model saw just the ~5 quarters Yahoo keeps. Each results
release filed on a 6-K has the quarter's revenue, gross profit, net income and share count, either in a statement
of operations table ("Three months ended", ASML) or in the release's opening sentences (TSMC). Every figure is in the
company's reporting currency; the model only uses ratios of them (growth, margins, per-share growth).
"""

import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from analysis.guidance import DEFAULT_AGENT, sec_get, text_of

FIELDS = ["Revenue", "Gross_Profit", "Net_Income", "Shares", "EPS"]
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
NUMBER = r"\(?-?[\d,]+(?:\.\d+)?\)?"
SCALES = {"thousand": 1e3, "thousands": 1e3, "million": 1e6, "millions": 1e6, "billion": 1e9, "billions": 1e9}
# Statement rows: the first label that matches is used. Later labels in a list are fallbacks.
ROWS = {
    "Revenue": [r"Total net sales", r"Total net revenues?", r"Total revenues?", r"Net sales", r"Net revenues?", r"Revenues?"],
    "Gross_Profit": [r"Gross profit", r"Gross margin"],
    "Net_Income": [r"Net income attributable to (?:equity holders|owners|shareholders)[^\d(]*", r"Net income", r"Net profit"],
    "EPS": [r"Basic net income per (?:ordinary )?share", r"Basic earnings per (?:ordinary )?share"],
}
QUARTER_WORDS = {"first": 3, "second": 6, "third": 9, "fourth": 12}


def result_documents(index_page):
    """The documents of a 6-K that can hold the results: its EX-99 exhibits and the 6-K itself (older releases, such
    as TSMC's before 2019, sit in the 6-K document)."""

    names = []
    for row in re.findall(r"(?is)<tr.*?</tr>", index_page):
        cells = re.findall(r"(?is)<td[^>]*>(.*?)</td>", row)
        if len(cells) >= 4 and re.fullmatch(r"\s*(EX-99(\.\d+)?|6-K)\s*", re.sub(r"<[^>]+>", "", cells[3])):
            link = re.search(r'href="([^"]+)"', cells[2])
            if link and link.group(1).lower().endswith((".htm", ".html", ".txt")):
                names.append(link.group(1).rsplit("/", 1)[-1])
    return names


def clean_text(text):
    """One line of text with negative numbers in brackets kept together: "( 900.3 )" becomes "(900.3)"."""

    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\(\s*([\d,]+(?:\.\d+)?)\s*\)", r"(\1)", text)


def _value(token):
    negative = token.startswith("(") or token.startswith("-")
    number = float(token.strip("()-").replace(",", ""))
    return -number if negative else number


def _quarter_end(year, month):
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def statement_quarters(text):
    """Quarters from a statement of operations table with "Three months ended" columns.

    The header names each column's month and day, then the years, e.g. "Three months ended Nine months ended Sep 29,
    Sep 28, Sep 29, Sep 28, (unaudited, in millions €, ...) 2024 2025 2024 2025". The first half of the columns are the
    three-month ones when a longer period follows. Returns {quarter end: {field: value}} for the three-month columns.
    """

    header = re.search(r"Three months ended(?P<longer>.{0,40}?(?:Six|Nine|Twelve) months ended)?(?P<body>.{0,400}?)"
                       r"(?P<years>(?:(?:19|20)\d\d\s+){2,8})", text, re.S | re.I)
    if not header:
        return {}
    years = [int(y) for y in header["years"].split()]
    days = re.findall(r"\b(" + "|".join(MONTHS) + r")[a-z]*\.?\s+(\d{1,2}),?", header["body"], re.I)
    unit = re.search(r"in (thousands|millions|billions)", text[header.start():header.end() + 300], re.I)
    scale = SCALES[unit[1].lower()] if unit else 1.0
    columns = len(years)
    quarter_columns = columns // 2 if header["longer"] and columns % 2 == 0 else columns
    if len(days) >= columns:
        ends = [_quarter_end(years[i], MONTHS[days[i][0][:3].lower()]) for i in range(quarter_columns)]
    else:
        return {}
    # A quarter that ends a few days into the next month (52/53-week years) belongs to the month before.
    ends = [end - pd.offsets.MonthEnd(1) if int(days[i][1]) <= 7 else end for i, end in enumerate(ends)]
    table = text[header.end():header.end() + 8000]
    out = {end: {} for end in ends}
    for field, labels in ROWS.items():
        for label in labels:
            row = re.search(r"(?<![A-Za-z])" + label + r"\s*(?:\d\s+)?((?:" + NUMBER + r"(?:\s+|$)){%d})" % columns, table, re.I)
            if row:
                values = [_value(token) for token in row[1].split()][:quarter_columns]
                for end, value in zip(ends, values):
                    out[end][field] = value if field == "EPS" else value * scale
                break
    shares = re.search(r"Weighted average number of[^:]{0,120}?:?\s*Basic\s+((?:" + NUMBER + r"(?:\s+|$)){%d})" % columns,
                       table, re.I) or re.search(
        r"Weighted average[^\d]{0,160}?(?:basic)?\s+((?:" + NUMBER + r"(?:\s+|$)){%d})" % columns, table, re.I)
    if shares:
        shares_unit = re.search(r"Weighted average[^:]{0,120}\(in (thousands|millions)\)", table, re.I)
        share_scale = SCALES[shares_unit[1].lower()] if shares_unit else scale
        for end, value in zip(ends, [_value(token) for token in shares[1].split()][:quarter_columns]):
            out[end]["Shares"] = value * share_scale
    return {end: figures for end, figures in out.items() if "Revenue" in figures}


def release_quarters(text):
    """One quarter from a release's opening sentence, e.g. "announced consolidated revenue of NT$1,494.24 billion,
    net income of NT$763.06 billion, and diluted earnings per share of NT$29.43 ... for the third quarter ended
    September 30, 2026", plus "Gross margin for the quarter was 66.4%"."""

    money = r"(?:NT\$|US\$|\$|€|EUR\s?|RMB\s?|¥|£)\s?([\d,]+(?:\.\d+)?)\s*(thousand|million|billion)?"
    revenue = re.search(r"(?:consolidated |total |net )?revenues? (?:of|was|were|totall?ed) " + money, text, re.I)
    period = re.search(r"(first|second|third|fourth) quarter (?:of \d{4} )?ended ([A-Z][a-z]+)\.? (\d{1,2}),? (\d{4})", text)
    if not revenue or not period:
        return {}
    end = _quarter_end(int(period[4]), MONTHS.get(period[2][:3].lower(), QUARTER_WORDS[period[1].lower()]))
    figures = {"Revenue": _value(revenue[1]) * SCALES.get((revenue[2] or "").lower(), 1.0)}
    income = re.search(r"net income (?:attributable to [^,]{0,60})?(?:of|was) " + money, text, re.I)
    if income:
        figures["Net_Income"] = _value(income[1]) * SCALES.get((income[2] or "").lower(), 1.0)
    eps = re.search(r"diluted (?:earnings|EPS) per (?:common )?share (?:of|was) " + money, text, re.I)
    if eps:
        figures["EPS"] = _value(eps[1])
        if "Net_Income" in figures and figures["EPS"] > 0:
            figures["Shares"] = figures["Net_Income"] / figures["EPS"]
    margin = re.search(r"Gross margin for the quarter was ([\d.]+)%", text, re.I)
    if margin:
        figures["Gross_Profit"] = figures["Revenue"] * float(margin[1]) / 100
    return {end: figures}


def results_quarters(text):
    """The quarters a results document reports: the statement table when there is one, else the opening sentence."""

    text = clean_text(text)
    return statement_quarters(text) or release_quarters(text)


def sixk_filings(submissions, session=None, agent=DEFAULT_AGENT):
    """Every 6-K in a company's submissions (including the older pages), as a frame of filed, accession."""

    pages = [submissions.get("filings", {}).get("recent", {})]
    for extra in submissions.get("filings", {}).get("files", []):
        if session is not None:
            response = sec_get("https://data.sec.gov/submissions/" + extra["name"], session, agent)
            if response is not None:
                pages.append(response.json())
    frames = [pd.DataFrame({"form": page.get("form", []), "filed": page.get("filingDate", []),
                            "accession": page.get("accessionNumber", [])}) for page in pages]
    filings = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["form", "filed", "accession"])
    return filings[filings["form"] == "6-K"].drop_duplicates("accession").reset_index(drop=True)


def near_releases(filings, release_dates, days=7):
    """The 6-Ks filed within ``days`` after a results release date (the release itself is usually filed that day)."""

    if not len(filings) or not len(release_dates):
        return filings.iloc[0:0]
    filed = pd.to_datetime(filings["filed"])
    releases = pd.DatetimeIndex(pd.to_datetime(release_dates)).normalize().sort_values()
    position = releases.searchsorted(filed, side="right") - 1
    gap = pd.Series([(f - releases[p]).days if p >= 0 else 99 for f, p in zip(filed, position)], index=filings.index)
    return filings[(gap >= 0) & (gap <= days)]


def filing_quarters(cik, accession, session, agent=DEFAULT_AGENT):
    """{quarter end: figures} from one 6-K's documents; the first document to give a field wins."""

    folder = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
    index = sec_get(folder + f"{accession}-index.htm", session, agent)
    found = {}
    if index is None:
        return found
    for name in result_documents(index.text):
        document = sec_get(folder + name, session, agent)
        if document is None:
            continue
        for end, figures in results_quarters(text_of(document.text)).items():
            found.setdefault(end, {})
            found[end].update({k: v for k, v in figures.items() if k not in found[end]})
    return found


def foreign_quarters(cik, release_dates, session, agent=DEFAULT_AGENT, since="2010-01-01"):
    """Quarterly Revenue, Gross_Profit, Net_Income and Shares by quarter end, from the 6-K results releases filed
    around ``release_dates``. The latest release wins where two report the same quarter (restated comparatives)."""

    response = sec_get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", session, agent)
    if response is None:
        return pd.DataFrame(columns=FIELDS)
    filings = sixk_filings(response.json(), session, agent)
    filings = near_releases(filings[filings["filed"] >= since], release_dates)
    filings = filings.sort_values("filed")
    # A few filings at a time: each is an index page and a few documents, and SEC allows 10 requests a second.
    with ThreadPoolExecutor(max_workers=3) as pool:
        found_list = list(pool.map(lambda accession: filing_quarters(cik, accession, session, agent), filings["accession"]))
    rows = {}
    for found in found_list:
        for end, figures in found.items():
            # A later release's comparative column is the restated figure: keep it, but never drop a field.
            rows[end] = {**rows.get(end, {}), **figures}
    if not rows:
        return pd.DataFrame(columns=FIELDS)
    frame = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=FIELDS).sort_index()
    # A quarter without a share count comes from an older layout the reader cannot date reliably (ASML before 2014).
    frame = frame[frame["Shares"].notna()]
    frame.index = pd.DatetimeIndex(frame.index)
    return frame.astype(float).replace(0, np.nan)


def align_units(other, reference, tolerance_days=14):
    """``other`` (e.g. Yahoo's quarters) with each column put on ``reference``'s units where they differ.

    Yahoo counts TSMC's shares in ADRs (one ADR = five shares) while the 6-K releases count ordinary shares; mixing
    them makes per-share growth jump fivefold. Over the quarters both have (dates within ``tolerance_days``), a
    column whose median ratio is off by more than 25% is scaled by that ratio.
    """

    if other.empty or reference.empty:
        return other
    other = other.copy()
    ref_index = pd.DatetimeIndex(reference.index)
    for column in [c for c in FIELDS if c in other and c in reference]:
        ratios = []
        for when, value in other[column].dropna().items():
            distance = np.abs(ref_index - pd.Timestamp(when))
            nearest = distance.argmin()
            match = reference[column].iloc[nearest]
            if distance[nearest] <= pd.Timedelta(days=tolerance_days) and pd.notna(match) and value:
                ratios.append(match / value)
        if ratios:
            ratio = float(np.median(ratios))
            if ratio > 0 and not 0.8 <= ratio <= 1.25:
                other[column] = other[column] * ratio
    return other
