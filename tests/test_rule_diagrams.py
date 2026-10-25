import pytest

from analysis.rule_diagrams import breakout_figure, day_breakout_figure, ma_rule_figure, pullback_figure, reversal_figure


def texts(fig):
    return " ".join(annotation.text for annotation in fig.layout.annotations)


@pytest.mark.parametrize("make", [ma_rule_figure, breakout_figure, pullback_figure, reversal_figure])
def test_every_rule_has_a_buy_illustration(make):
    fig = make()
    assert "BUY" in texts(fig)
    assert fig.layout.title.text == "How the rule trades (illustration)"


def test_labels_follow_the_settings():
    assert "30%" in texts(day_breakout_figure(10, "top_range", 0.7))
    assert "Prior 15-day high" in texts(day_breakout_figure(15, "same_day"))
    assert "SMA 20" in texts(day_breakout_figure(10, "above_ma", ma_label="SMA 20"))
    assert "yesterday's close" in texts(day_breakout_figure(10, "daily_break"))
    assert "prior 5-day<br>average close" in texts(day_breakout_figure(10, "daily_break", break_days=5))
    assert "Prior 55-bar high" in texts(breakout_figure(55, 2.0)) and "2x" in texts(breakout_figure(55, 2.0))
    assert "EMA 9" in texts(ma_rule_figure("EMA 9", "SMA 30"))


def test_tuned_exits_are_labelled():
    assert "SMA" in texts(breakout_figure(20, 1.5, 150)) and "150-bar SMA" in texts(breakout_figure(20, 1.5, 150))
    assert "after 7 bars" in texts(reversal_figure(20, 50, 200, 7))


def test_pullback_shows_every_setting():
    text = texts(pullback_figure(21, 40, 150, 2.0, 30, 7))
    for label in ["Fast MA 40", "Slow MA 150", "Pullback EMA 21", "lookback: 30 bars", "last 7 bars", "2 ATR"]:
        assert label in text, label
