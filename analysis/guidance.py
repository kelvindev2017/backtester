"""Company guidance extraction: what a company said it expects (revenue, revenue growth, EPS), read from the
press releases it files with the SEC (8-K Items 2.02 results and 7.01 updates, exhibits EX-99.x).

The module is separate from the Valuation Model. Its output is a table in the model's guidance interface
(see ``valuation_model.GUIDANCE_COLUMNS``): one row per guided figure, dated by the filing day, so the model can
use what was known on each day. Extraction is rule-based (no language model): it reads the outlook sentences and
guidance-table lines, keeps the ones that look forward, and pulls the first figure after the revenue, sales or
EPS word. What it cannot read it leaves out; the model then holds the last reported quarter.

Coverage limits: foreign filers report on 6-K (not read), and companies that guide only on the call (e.g. Apple)
leave nothing in the release.
"""

import html
import re
import time

import pandas as pd

from analysis.valuation_model import GUIDANCE_COLUMNS  # the model's guidance interface

DEFAULT_AGENT = "StreamlitBacktester/1.0 kelvin@example.com"  # same default as App.py; SEC asks for an email form

GUIDANCE_WORDS = re.compile(r"\b(outlook|guidance|expects?|expected|anticipates?|forecasts?|now estimates?|"
                            r"estimates? (?:net |total )?(?:sales|revenue|earnings|eps)|reaffirm\w*|"
                            r"(?:raises?|raised|lowers?|lowered|narrows?|narrowed|updates?|updated) (?:its |the )?"
                            r"(?:full[- ]year |fiscal )?(?:outlook|guidance|forecast))\b", re.I)
NUMBER = re.compile(r"(\$\s?\d[\d,.]*|\d[\d,.]*\s?(?:%|percent|billion|million))", re.I)
FORWARD = re.compile(r"\b(expect\w*|anticipat\w*|outlook|forecast\w*|estimat\w*|projects?|projected|range of|"
                     r"reaffirm\w*|sees|plus or minus|to be (?:approximately|in|between|about|around))\b", re.I)
GUIDE = re.compile(r"\b(guidance|outlook|approximately|at least|between)\b", re.I)
PAST = re.compile(r"\b(was|were|reported|reporte d|increased|decreased|rose|fell|grew|declined|totaled|achieved|"
                  r"deliver\w*|impacted|included|record)\b", re.I)
REVENUE = re.compile(r"\b(total company revenues?|total revenues?|net revenues?|revenues?|total net sales|"
                     r"net sales|total sales|sales)\b", re.I)
EPS = re.compile(r"\b(EPS|earnings per (?:diluted )?share|diluted earnings per share|per diluted share)\b", re.I)
NOT_REVENUE = re.compile(r"\b(comparable|same[- ]store|comp(?:arable)? store|cost of (?:revenue|sales)|"
                         r"revenue recognition|deferred revenue|recurring revenue|gross margin|"
                         r"(?:non-\w+|cloud|segment|services?|products?|subscriptions?|division|business|data center|"
                         r"automotive|handset|iot|software|hardware|license|licensing) (?:net )?(?:revenues?|sales))\b",
                         re.I)
QUARTER = re.compile(r"\b(first|second|third|fourth|1st|2nd|3rd|4th|next|current) (?:fiscal |calendar )?quarter|"
                     r"\bQ[1-4]\b|\bquarter\b", re.I)
YEAR = re.compile(r"\b(full[- ]year|fiscal(?: year)? ?(?:20)?\d{2}\b|fy ?(?:20)?\d{2}\b|for (?:the )?(?:fiscal )?year|"
                  r"annual|20\d\d (?:revenue|net sales|sales|eps|guidance|outlook|financial guidance)|"
                  r"(?:for|in) 20\d\d)", re.I)
FULL_YEAR = re.compile(r"\b(full[- ]year|for the (?:fiscal )?year|annual|entire year|fy ?(?:20)?\d{2} (?:outlook|guidance)|"
                       r"fiscal (?:year )?(?:20)?\d{2} (?:outlook|guidance))\b", re.I)
UNIT = {"billion": 1e9, "b": 1e9, "bn": 1e9, "million": 1e6, "m": 1e6, "mm": 1e6}
AMOUNT = r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(billion|million|bn|mm|b|m)?\b"
AMOUNT_RANGE = re.compile(AMOUNT + r"\s*(?:to|-|–|—|and)\s*\$?\s?(\d[\d,]*(?:\.\d+)?)\s*(billion|million|bn|mm|b|m)?\b",
                          re.I)
AMOUNT_ONE = re.compile(AMOUNT + r"(?:,?\s*(?:plus or minus|\+/-|±)\s*(\d+(?:\.\d+)?)\s*%)?", re.I)
PERCENT = r"\(?(-?\d+(?:\.\d+)?)\)?\s*(?:%|percent)"
PERCENT_RANGE = re.compile(r"\(?(-?\d+(?:\.\d+)?)\)?\s*(?:%|percent)?\s*(?:to|-|–|—)\s*" + PERCENT, re.I)
PERCENT_ONE = re.compile(PERCENT, re.I)


def _number(text):
    return float(text.replace(",", ""))


def _scale(unit, other_unit=None):
    unit = (unit or other_unit or "").lower()
    return UNIT.get(unit)


def _clean(sentence):
    """Split ``[heading] sentence`` into the heading and the sentence."""

    match = re.match(r"\[(.*?)\]\s*(.*)", sentence)
    return (match.group(1), match.group(2)) if match else ("", sentence)


def guidance_sentences(text):
    """Sentences stating an outlook with a number, plus the numeric lines of an "Outlook" or "Guidance" section
    (many releases give guidance as a table), prefixed with that section's heading."""

    sentences = re.split(r"(?<=[.;])\s+(?=[A-Z(•])|\n+", text)
    kept, heading, left = [], "", 0
    for sentence in sentences:
        sentence = re.sub(r"\s+", " ", sentence).strip(" •-")
        if not sentence:
            continue
        if len(sentence) <= 80 and re.search(r"\b(outlook|guidance)\b", sentence, re.I) \
                and not re.search(r"\b(reports?|results|raises|lowers)\b", sentence, re.I):
            heading, left = sentence, 25
            continue
        if left and re.match(r"(about |conference call|forward-looking|notes?\b|non-gaap financial measures)", sentence, re.I):
            left = 0
        if re.search(r"forward-looking statements|safe harbor|undue reliance|not able to reconcile", sentence, re.I):
            continue
        if left:
            left -= 1
            if 8 <= len(sentence) <= 700 and NUMBER.search(sentence):
                kept.append(f"[{heading}] {sentence}")
                continue
        if 40 <= len(sentence) <= 700 and GUIDANCE_WORDS.search(sentence) and NUMBER.search(sentence):
            kept.append(sentence)
    return kept


def sentence_period(heading, sentence, default_quarter=True):
    """"quarter", "year" or None (unclear): the sentence decides, else its heading, else a plain outlook
    (no period named) is the next quarter, as most quarterly outlooks are written."""

    for text in (sentence, heading):
        quarter, year = QUARTER.search(text), YEAR.search(text)
        whole_year = FULL_YEAR.search(text)
        if quarter and not whole_year:
            return "quarter"  # "Q1 FY27", "third quarter of fiscal 2026": a quarter of that year
        if year and not quarter:
            return "year"
        if quarter and whole_year:
            if text is sentence:
                # "full year and fourth quarter" in one sentence: the period word named first decides.
                return "year" if whole_year.start() < quarter.start() else "quarter"
            return None
    return "quarter" if default_quarter else None


def _window(sentence, anchor):
    """Text after the anchor word, up to the next metric word, so a figure is read from its own clause."""

    rest = sentence[anchor.end():]
    stop = re.search(r"\b(gross margin|operating (?:margin|expenses?|income)|tax rate|EPS|earnings per|"
                     r"cost of|capital expenditures|free cash flow|segment|revenues?|net sales)\b", rest, re.I)
    return rest[: min(stop.start() if stop else 220, 220)]


def _basis(text):
    lowered = text.lower()
    for word in ("organic", "operational", "constant currency", "adjusted", "non-gaap", "gaap", "reported"):
        if word in lowered:
            return word
    return ""


def _revenue_figure(window, hint=None):
    """``(metric, low, high)`` from the clause after a revenue word, or None. A table line with two ranges
    (previous and updated guidance) gives the last one. ``hint`` is the unit a table states once ("($B)")."""

    ranges = list(AMOUNT_RANGE.finditer(window))
    if ranges:
        # Two ranges side by side in a table line are the previous and the updated guidance.
        match = ranges[1] if len(ranges) >= 2 and ranges[1].start() - ranges[0].end() <= 3 else ranges[0]
        scale = _scale(match.group(2), match.group(4)) or hint
        if scale:
            return "revenue", _number(match.group(1)) * scale, _number(match.group(3)) * scale
    one = AMOUNT_ONE.search(window)
    if one and (_scale(one.group(2)) or hint):
        value = _number(one.group(1)) * (_scale(one.group(2)) or hint)
        spread = float(one.group(3)) / 100 if one.group(3) else 0.0
        return "revenue", value * (1 - spread), value * (1 + spread)
    if re.search(r"\b(growth|grow|increase|up|decline|down|change)\b", window, re.I) and "$" not in window[:60]:
        sign = -1.0 if re.search(r"\b(decline|decrease|down)\b", window, re.I) else 1.0
        found = list(PERCENT_RANGE.finditer(window))
        if found:
            match = found[1] if len(found) >= 2 and found[1].start() - found[0].end() <= 3 else found[0]
            low, high = sorted([sign * _number(match.group(1)), sign * _number(match.group(2))])
            return "revenue_growth", low, high
        one = PERCENT_ONE.search(window)
        if one:
            value = sign * _number(one.group(1))
            if "(" in window[max(0, one.start() - 1): one.start() + 1]:
                value = -abs(value)
            return "revenue_growth", value, value
    return None


def _eps_figure(window):
    ranges = list(AMOUNT_RANGE.finditer(window))
    if ranges:
        match = ranges[1] if len(ranges) >= 2 and ranges[1].start() - ranges[0].end() <= 3 else ranges[0]
        if not _scale(match.group(2), match.group(4)):
            return "eps", _number(match.group(1)), _number(match.group(3))
    one = re.search(r"\$\s?(\d+(?:\.\d+)?)(?!\s*(?:billion|million|bn|mm|b|m)\b)", window, re.I)
    if one:
        return "eps", _number(one.group(1)), _number(one.group(1))
    return None


def extract_figures(sentence):
    """Guided figures in one outlook sentence: a list of dicts with period, metric, low, high, basis."""

    heading, body = _clean(sentence)
    if not (FORWARD.search(body) or GUIDE.search(body) or re.search(r"\b(outlook|guidance)\b", heading, re.I)):
        return []
    if PAST.search(body) and not FORWARD.search(body):
        return []  # a result, even if it mentions guidance ("revenues at the high end of our guidance")
    if re.search(r"compared (?:to|with) (?:the |our )?(?:company's )?(?:prior |forward-looking )?guidance|"
                 r"versus (?:the )?guidance|guidance range of .* compared", body, re.I):
        return []
    period = sentence_period(heading, body)
    if period is None:
        return []
    figures = []
    anchors = [m for m in REVENUE.finditer(body) if not NOT_REVENUE.search(body[max(0, m.start() - 25): m.end() + 15])]
    total = [m for m in anchors if re.match(r"total|net", m.group(1), re.I)]
    for anchor in (total or anchors)[:1]:
        hint = re.search(r"\(\$\s?(B|M)\)|\$ in (billions|millions)|in (billions|millions)", heading + " " + body, re.I)
        hint = _scale(next(g for g in hint.groups() if g)[0]) if hint else None
        found = _revenue_figure(_window(body, anchor), hint)
        if found:
            metric, low, high = found
            if metric == "revenue_growth" and re.search(r"\b(organic|operational|constant currency)\b", body, re.I) \
                    and re.search(r"\breported\b", body, re.I):
                # "x percent on a reported basis and y percent on an organic basis": the reported figure.
                reported = re.search(r"(-?\d+(?:\.\d+)?)(?:\s*(?:%|percent)?\s*(?:to|-|–)\s*(-?\d+(?:\.\d+)?))?"
                                     r"\s*(?:%|percent)[^.;%]{0,40}\breported\b", body, re.I)
                if reported:
                    low = _number(reported.group(1))
                    high = _number(reported.group(2)) if reported.group(2) else low
            figures.append({"period": period, "metric": metric, "low": low, "high": high,
                            "basis": _basis(body[anchor.start(): anchor.start() + 160])})
    eps = EPS.search(body)
    if eps:
        # The figure after the EPS word, before any "impact of", "includes", "an increase of" clause.
        after = re.split(r"\b(?:by|impacts?|includes?|excludes?|benefits?|increases?|decreases?|reflects?|offset)\b",
                         body[eps.end(): eps.end() + 120], maxsplit=1, flags=re.I)[0]
        found = _eps_figure(after)
        if found and 0 < abs(found[1]) < 1000:
            # An unclear period is settled by the filing's revenue guidance (see extract_filing).
            figures.append({"period": sentence_period(heading, body, default_quarter=False), "metric": "eps",
                            "low": found[1], "high": found[2],
                            "basis": _basis(body[max(0, eps.start() - 60): eps.end() + 60])})
    for figure in figures:
        if figure["low"] > figure["high"]:
            figure["low"], figure["high"] = figure["high"], figure["low"]
    # A range whose ends are far apart is a misread (two different figures), not guidance.
    return [f for f in figures if f["metric"] == "revenue_growth" or f["low"] <= 0 or f["high"] <= 1.5 * f["low"]]


def fiscal_period_end(filed, period, fiscal_year_end):
    """The end of the guided period: the first fiscal quarter end after the filing day, or the end of the fiscal
    year that quarter belongs to. ``fiscal_year_end`` is the SEC's "MMDD" (e.g. "1231")."""

    filed = pd.Timestamp(filed)
    try:
        month, day = int(str(fiscal_year_end)[:2]), int(str(fiscal_year_end)[2:4])
    except (TypeError, ValueError):
        month, day = 12, 31
    ends = []
    for year in range(filed.year - 1, filed.year + 3):
        try:
            year_end = pd.Timestamp(year=year, month=month, day=min(day, 28) if month == 2 else day)
        except ValueError:
            year_end = pd.Timestamp(year=year, month=month, day=28)
        ends += [(year_end - pd.DateOffset(months=3 * k), year_end) for k in range(4)]
    ends = sorted(end for end in ends if end[0] > filed + pd.Timedelta(days=5))
    if not ends:
        return pd.NaT
    quarter_end, year_end = ends[0]
    return quarter_end if period == "quarter" else year_end


def extract_filing(text, filed, fiscal_year_end, ticker="", accession=""):
    """Guidance rows (``GUIDANCE_COLUMNS``) from one press release's text; one row per period and metric, the
    first found (the headline figure)."""

    figures = [dict(figure, text=sentence) for sentence in guidance_sentences(text)
               for figure in extract_figures(sentence)]
    revenue_periods = {f["period"] for f in figures if f["metric"] != "eps" and f["period"]}
    rows, seen = [], set()
    for figure in figures:
        if figure["period"] is None:  # an EPS line with no period: the period of the filing's revenue guidance
            if len(revenue_periods) != 1:
                continue
            figure["period"] = next(iter(revenue_periods))
        key = (figure["period"], figure["metric"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({"date": pd.Timestamp(filed).normalize(), "ticker": ticker, "accession": accession,
                     "period": figure["period"],
                     "period_end": fiscal_period_end(filed, figure["period"], fiscal_year_end),
                     "metric": figure["metric"], "low": figure["low"], "high": figure["high"],
                     "basis": figure["basis"], "text": figure["text"][:400]})
    return rows


# --- SEC access -------------------------------------------------------------------------------------------------

def sec_get(url, session, agent=DEFAULT_AGENT):
    for attempt in range(4):
        try:
            response = session.get(url, headers={"User-Agent": agent, "Accept-Encoding": "gzip, deflate"}, timeout=30)
        except Exception:
            time.sleep(2 * (attempt + 1))
            continue
        time.sleep(0.12)  # SEC asks for at most 10 requests a second
        if response.status_code == 200:
            return response
        if response.status_code in (403, 429, 500, 502, 503):
            time.sleep(2 * (attempt + 1))
            continue
        return None
    return None


def exhibit_names(index_page):
    """Press release exhibits (EX-99.x) listed in a filing's index page; companies name the files freely."""

    names = []
    for row in re.findall(r"(?is)<tr.*?</tr>", index_page):
        cells = re.findall(r"(?is)<td[^>]*>(.*?)</td>", row)
        if len(cells) >= 4 and re.fullmatch(r"\s*EX-99(\.\d+)?\s*", re.sub(r"<[^>]+>", "", cells[3])):
            link = re.search(r'href="([^"]+)"', cells[2])
            if link and link.group(1).lower().endswith((".htm", ".html", ".txt")):
                names.append(link.group(1).rsplit("/", 1)[-1])
    return names


def text_of(document):
    document = re.sub(r"(?is)<(script|style).*?</\1>", " ", document)
    document = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>|</li>", "\n", document)
    document = html.unescape(re.sub(r"<[^>]+>", " ", document))
    return re.sub(r"[ \t\xa0]+", " ", document)


def release_filings(submissions, session=None, agent=DEFAULT_AGENT, since="2000-01-01"):
    """8-K filings with Item 2.02 or 7.01 since a date, from the submissions index and its older pages:
    a frame of form, filed, accession, items."""

    pages = [submissions.get("filings", {}).get("recent", {})]
    for older in submissions.get("filings", {}).get("files", []) or []:
        if older.get("filingTo", "9999") < since or session is None:
            continue
        response = sec_get(f"https://data.sec.gov/submissions/{older['name']}", session, agent)
        if response is not None:
            pages.append(response.json())
    frames = []
    for page in pages:
        frames.append(pd.DataFrame({"form": page.get("form", []), "filed": page.get("filingDate", []),
                                    "accession": page.get("accessionNumber", []), "items": page.get("items", [])}))
    filings = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["form", "filed", "accession", "items"])
    items = filings["items"].fillna("").astype(str)
    keep = (filings["form"] == "8-K") & (filings["filed"] >= since) & (items.str.contains("2.02") | items.str.contains("7.01"))
    return filings[keep].drop_duplicates("accession").sort_values("filed").reset_index(drop=True)


def day_filings(day, session, agent=DEFAULT_AGENT):
    """8-K filings made on one day, from the SEC's daily index (one request for the whole market): a frame of
    cik, accession, filed. None while the day's index is not published yet (or the day had no filings)."""

    day = pd.Timestamp(day)
    quarter = (day.month - 1) // 3 + 1
    url = f"https://www.sec.gov/Archives/edgar/daily-index/{day.year}/QTR{quarter}/form.{day:%Y%m%d}.idx"
    response = sec_get(url, session, agent)
    if response is None:
        return None
    rows = []
    for line in response.text.splitlines():
        if not line.startswith("8-K "):
            continue
        found = re.search(r"\s(\d{3,10})\s+(\d{8})\s+edgar/data/\d+/([\d-]+)\.txt", line)
        if found:
            rows.append({"cik": int(found.group(1)), "accession": found.group(3),
                         "filed": f"{found.group(2)[:4]}-{found.group(2)[4:6]}-{found.group(2)[6:]}"})
    return pd.DataFrame(rows, columns=["cik", "accession", "filed"])


def results_or_update(index_page):
    """Whether a filing's index page lists Item 2.02 (results) or 7.01 (an update such as new guidance)."""

    return bool(re.search(r"Item\s+(2\.02|7\.01)", index_page))


def filing_guidance(cik, accession, filed, fiscal_year_end, session, ticker="", agent=DEFAULT_AGENT, items_check=False):
    """Guidance rows of one 8-K, read from its EX-99 exhibits (results first, so the headline figure wins). With
    ``items_check``, a filing that is neither results nor an update (Item 2.02 / 7.01) is skipped."""

    folder = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"
    index = sec_get(f"{folder}/{accession}-index.html", session, agent)
    if index is None or (items_check and not results_or_update(index.text)):
        return []
    rows = []
    for name in exhibit_names(index.text):
        document = sec_get(f"{folder}/{name}", session, agent)
        if document is not None:
            rows += extract_filing(text_of(document.text), filed, fiscal_year_end, ticker, accession)
    unique, seen = [], set()
    for row in rows:
        if (row["period"], row["metric"]) not in seen:
            seen.add((row["period"], row["metric"]))
            unique.append(row)
    return unique


def company_guidance(ticker, cik, session, agent=DEFAULT_AGENT, since="2010-01-01", skip=()):
    """All guidance rows a company's 8-K releases give since a date (accessions in ``skip`` are not read again).
    Returns ``(rows, accessions read)``."""

    submissions = sec_get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", session, agent)
    if submissions is None:
        return [], []
    submissions = submissions.json()
    fiscal_year_end = submissions.get("fiscalYearEnd") or "1231"
    rows, read = [], []
    for filing in release_filings(submissions, session, agent, since).itertuples():
        if filing.accession in skip:
            continue
        rows += filing_guidance(cik, filing.accession, filing.filed, fiscal_year_end, session, ticker, agent)
        read.append(filing.accession)
    return rows, read
