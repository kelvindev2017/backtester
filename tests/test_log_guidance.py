import analysis.guidance as guidance


def test_outlook_sentences_and_guidance_tables_are_kept():
    text = ("Boston Scientific generated net sales of $5.1 billion. The company now estimates net sales growth for the "
            "full year 2026 to be approximately 4.25 percent on a reported basis.\n"
            "Operating expenses fell 3% on lower travel.\n"
            "This release contains forward-looking statements; actual results may differ by 10%.\n"
            "Full Year 2026 Guidance\nOrganic revenues At least 5% growth\nComparable EPS growth 8% to 9%\n"
            "About the company\nFounded in 1886 with 700,000 employees.")
    kept = guidance.guidance_sentences(guidance.text_of(text))
    assert any("now estimates net sales growth" in s for s in kept)
    assert "[Full Year 2026 Guidance] Organic revenues At least 5% growth" in kept
    assert "[Full Year 2026 Guidance] Comparable EPS growth 8% to 9%" in kept
    assert not any("travel" in s or "forward-looking" in s or "1886" in s for s in kept)


def test_press_release_exhibits_are_found_by_type_not_file_name():
    page = ('<table><tr><td>1</td><td>8-K</td><td><a href="/Archives/x/bsx-20261028.htm">bsx</a></td><td>8-K</td></tr>'
            '<tr><td>2</td><td>Release</td><td><a href="/Archives/x/earningsrelease_live.htm">r</a></td>'
            '<td>EX-99.1</td></tr><tr><td>3</td><td>Img</td><td><a href="/Archives/x/a.jpg">i</a></td>'
            '<td>GRAPHIC</td></tr></table>')
    assert guidance.exhibit_names(page) == ["earningsrelease_live.htm"]


class Page:
    def __init__(self, text):
        self.text = text


def test_the_days_8k_filings_come_from_the_daily_list(monkeypatch):
    listing = ("Form Type   Company Name   CIK   Date Filed  File Name\n" + "-" * 40 + "\n"
               "8-K              BOSTON SCIENTIFIC CORP          885725      20261028    "
               "edgar/data/885725/0000885725-26-000123.txt\n"
               "8-K/A            OTHER CO                        111         20261028    edgar/data/111/0000111-26-1.txt\n"
               "10-Q             BOSTON SCIENTIFIC CORP          885725      20261028    "
               "edgar/data/885725/0000885725-26-000124.txt\n")
    asked = []
    monkeypatch.setattr(guidance, "sec_get", lambda url, session, agent=None: asked.append(url) or Page(listing))
    filings = guidance.day_filings("2026-10-28", None)
    assert asked == ["https://www.sec.gov/Archives/edgar/daily-index/2026/QTR4/form.20261028.idx"]
    assert filings.to_dict("records") == [{"cik": 885725, "accession": "0000885725-26-000123", "filed": "2026-10-28"}]
    monkeypatch.setattr(guidance, "sec_get", lambda url, session, agent=None: None)
    assert guidance.day_filings("2026-10-29", None) is None  # not published yet


def test_only_results_and_updates_are_opened():
    assert guidance.results_or_update('<div class="info">Item 2.02: Results of Operations</div>')
    assert guidance.results_or_update("Item 7.01: Regulation FD Disclosure")
    assert not guidance.results_or_update("Item 5.02: Departure of Directors")
