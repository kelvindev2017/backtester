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
    assert "Prior 55-bar high" in texts(breakout_figure(55, 2.0)) and "2x" in texts(breakout_figure(55, 2.0))
    assert "EMA 9" in texts(ma_rule_figure("EMA 9", "SMA 30"))
