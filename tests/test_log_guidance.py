import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "log_guidance", os.path.join(os.path.dirname(__file__), "..", "scripts", "log_guidance.py"))
log_guidance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(log_guidance)


def test_outlook_sentences_and_guidance_tables_are_kept():
    text = ("Boston Scientific generated net sales of $5.1 billion. The company now estimates net sales growth for the "
            "full year 2026 to be approximately 4.25 percent on a reported basis.\n"
            "Operating expenses fell 3% on lower travel.\n"
            "This release contains forward-looking statements; actual results may differ by 10%.\n"
            "Full Year 2026 Guidance\nOrganic revenues At least 5% growth\nComparable EPS growth 8% to 9%\n"
            "About the company\nFounded in 1886 with 700,000 employees.")
    kept = log_guidance.guidance_sentences(log_guidance.text_of(text))
    assert any("now estimates net sales growth" in s for s in kept)
    assert "[Full Year 2026 Guidance] Organic revenues At least 5% growth" in kept
    assert "[Full Year 2026 Guidance] Comparable EPS growth 8% to 9%" in kept
    assert not any("travel" in s or "forward-looking" in s or "1886" in s for s in kept)


def test_press_release_exhibits_are_found_by_type_not_file_name():
    page = ('<table><tr><td>1</td><td>8-K</td><td><a href="/Archives/x/bsx-20261028.htm">bsx</a></td><td>8-K</td></tr>'
            '<tr><td>2</td><td>Release</td><td><a href="/Archives/x/earningsrelease_live.htm">r</a></td>'
            '<td>EX-99.1</td></tr><tr><td>3</td><td>Img</td><td><a href="/Archives/x/a.jpg">i</a></td>'
            '<td>GRAPHIC</td></tr></table>')
    assert log_guidance.exhibit_names(page) == ["earningsrelease_live.htm"]
