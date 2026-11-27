import numpy as np
import pandas as pd
import pytest

from analysis.foreign_results import align_units, near_releases, results_quarters

ASML_TABLE = """Consolidated Statements of Operations Three months ended, Nine months ended, Sep 29, Sep 28, Sep 29,
Sep 28, (unaudited, in millions €, except per share data) 2024 2025 2024 2025 Net system sales 5,926.0 5,553.8
14,652.8 16,890.3 Net service and field option sales 1,541.3 1,962.2 4,347.3 6,058.9 Total net sales 7,467.3 7,516.0
19,000.1 22,949.2 Total cost of sales ( 3,673.9 ) (3,635.7) (9,297.9) (10,759.7) Gross profit 3,793.4 3,880.3 9,702.2
12,189.5 Net income 2,076.5 2,124.5 4,878.2 6,769.8 Basic net income per ordinary share 5.28 5.49 12.40 17.39
Weighted average number of ordinary shares used in computing per share amounts (in millions): Basic 393.2 387.3
393.3 389.2 Diluted 393.6 387.6 393.6 389.5"""

# Older layout: the unit comes after the years and a 52/53-week quarter ends a few days into the next month.
ASML_2016 = """Three months ended, Six months ended, Jun 28, Jul 3, Jun 28, Jul 3, 2015 2016 2015 2016 (in millions EUR,
except per share data) Total net sales 1,654.1 1,739.6 3,304.0 3,072.8 Gross profit 753.8 741.4 1,532.4 1,309.5
Net income 369.7 353.8 772.4 551.8 Weighted average number of ordinary shares used in computing per share amounts
(in millions): Basic 431.4 424.5 432.0 425.7"""

TSM_RELEASE = """TSMC Reports Third Quarter EPS of NT$29.43 HSINCHU, Taiwan, R.O.C., Oct. 15, 2026 -- TSMC today announced
consolidated revenue of NT$1,494.24 billion, net income of NT$763.06 billion, and diluted earnings per share of
NT$29.43 (US$4.60 per ADR unit) for the third quarter ended September 30, 2026. Gross margin for the quarter was
66.4%, operating margin was 58.7%."""


def test_statement_table_gives_the_three_month_columns():
    quarters = results_quarters(ASML_TABLE)
    assert sorted(quarters) == [pd.Timestamp("2024-09-30"), pd.Timestamp("2025-09-30")]
    latest = quarters[pd.Timestamp("2025-09-30")]
    assert latest["Revenue"] == pytest.approx(7516.0e6)
    assert latest["Gross_Profit"] == pytest.approx(3880.3e6)
    assert latest["Net_Income"] == pytest.approx(2124.5e6)
    assert latest["Shares"] == pytest.approx(387.3e6)
    assert latest["EPS"] == pytest.approx(5.49)
    assert quarters[pd.Timestamp("2024-09-30")]["Revenue"] == pytest.approx(7467.3e6)


def test_older_layout_unit_after_years_and_week_based_quarter_end():
    quarters = results_quarters(ASML_2016)
    assert sorted(quarters) == [pd.Timestamp("2015-06-30"), pd.Timestamp("2016-06-30")]
    assert quarters[pd.Timestamp("2016-06-30")]["Revenue"] == pytest.approx(1739.6e6)
    assert quarters[pd.Timestamp("2016-06-30")]["Shares"] == pytest.approx(424.5e6)


def test_release_sentence_gives_the_quarter():
    quarters = results_quarters(TSM_RELEASE)
    figures = quarters[pd.Timestamp("2026-09-30")]
    assert figures["Revenue"] == pytest.approx(1494.24e9)
    assert figures["Net_Income"] == pytest.approx(763.06e9)
    assert figures["Gross_Profit"] == pytest.approx(1494.24e9 * 0.664)
    assert figures["Shares"] == pytest.approx(763.06e9 / 29.43)


def test_no_results_in_other_documents():
    assert results_quarters("The board approved a dividend of NT$6 per share.") == {}


def test_only_six_ks_filed_just_after_a_release():
    filings = pd.DataFrame({"form": "6-K", "filed": ["2026-10-14", "2026-10-20", "2026-11-25", "2026-10-10"],
                            "accession": list("abcd")})
    kept = near_releases(filings, pd.DatetimeIndex(["2026-10-14 05:00"]))
    assert kept["accession"].tolist() == ["a", "b"]


def test_yahoo_adr_share_counts_put_on_ordinary_shares():
    reference = pd.DataFrame({"Revenue": [1.0e12, 1.1e12], "Shares": [25.93e9, 25.93e9]},
                             index=pd.DatetimeIndex(["2025-09-30", "2025-12-31"]))
    yahoo = pd.DataFrame({"Revenue": [1.0e12, 1.1e12, 1.2e12], "Shares": [5.186e9, 5.186e9, 5.186e9]},
                         index=pd.DatetimeIndex(["2025-09-30", "2025-12-31", "2026-03-31"]))
    aligned = align_units(yahoo, reference)
    assert aligned["Shares"].iloc[-1] == pytest.approx(25.93e9, rel=1e-3)
    assert aligned["Revenue"].tolist() == yahoo["Revenue"].tolist()
    assert align_units(pd.DataFrame(), reference).empty
    assert np.isnan(align_units(yahoo.assign(Shares=np.nan), reference)["Shares"]).all()


def test_log_round_trip_and_restated_quarter_replaces_old_row():
    from analysis.foreign_results import from_log, merge_log, to_log

    first = pd.DataFrame({"Revenue": [1.0, 2.0], "Shares": [10.0, 10.0]},
                         index=pd.DatetimeIndex(["2026-03-31", "2026-06-30"]))
    log = merge_log(pd.DataFrame(), to_log("ASML", first))
    restated = pd.DataFrame({"Revenue": [2.5], "Shares": [10.0]}, index=pd.DatetimeIndex(["2026-06-30"]))
    log = merge_log(log, to_log("ASML", restated))
    back = from_log(log, "ASML")
    assert back["Revenue"].tolist() == [1.0, 2.5]
    assert from_log(log, "TSM").empty and from_log(pd.DataFrame(), "ASML").empty


def test_a_read_past_its_deadline_raises_instead_of_returning_part(monkeypatch):
    import analysis.foreign_results as F

    class Reply:
        def json(self):
            return {"filings": {"recent": {"form": ["6-K"], "filingDate": ["2026-10-14"], "accessionNumber": ["a"]}}}

    monkeypatch.setattr(F, "sec_get", lambda *args, **kwargs: Reply())
    with pytest.raises(TimeoutError):
        F.foreign_quarters(1, None, None, since="2026-01-01", deadline=0)


def test_daily_run_adds_foreign_quarters(tmp_path, monkeypatch):
    import sys
    from datetime import date

    sys.path.insert(0, "scripts")
    import log_guidance as L

    monkeypatch.setattr(L, "sec_ciks", lambda session: {"ASML": 937966})
    calls = []

    def fake(cik, releases, session, agent, since):
        calls.append(since)
        return pd.DataFrame({"Revenue": [11.66e9], "Shares": [383e6]}, index=pd.DatetimeIndex(["2026-09-30"]))

    monkeypatch.setattr(L, "foreign_quarters", fake)
    dates = lambda ticker: pd.DatetimeIndex(["2026-10-14"])
    L.update_foreign(str(tmp_path), ["ASML", "NOPE"], None, date(2026, 11, 27), dates)
    L.update_foreign(str(tmp_path), ["ASML"], None, date(2026, 11, 27), dates)
    log = pd.read_csv(tmp_path / "foreign_quarters.csv")
    assert log["ticker"].tolist() == ["ASML"] and log["Revenue"].iloc[0] == pytest.approx(11.66e9)
    # The history once, then only the last 20 days.
    assert calls == [L.FOREIGN_SINCE, "2026-11-07"]
    assert pd.read_csv(tmp_path / "foreign_read.csv")["ticker"].tolist() == ["ASML"]
