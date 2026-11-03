"""Charts zoom along x only, and each y-axis fits the data in view (the browser side is CHART_FIT_Y_JS in App.py).

``fit_y_axes`` turns off the vertical zoom of every y-axis it can fit (``fixedrange``) and lists those axes in
``layout.meta["fit_y"]``, so the page script knows which axes to rescale after a zoom or pan of the dates:

- ``{"fit": True}``: fit the lowest and highest value in view, with a little room above and below;
- ``{"cap": [low, high]}``: the chart set this range itself (e.g. a P/E or PEG axis capped against outliers), so
  the fitted range stays inside it, and the chart opens on it;
- ``{"volume": True}``: bars on an overlay axis (volume under the price), kept to the bottom quarter of the panel.
"""

FIT_TRACE_TYPES = {"scatter", "scattergl", "bar", "candlestick", "ohlc"}


def _axis_key(short_name, letter):
    """Layout key of a trace's axis: "y" -> "yaxis", "y3" -> "yaxis3"."""

    return short_name.replace(letter, f"{letter}axis", 1) if short_name else f"{letter}axis"


def fit_y_axes(fig):
    """Make ``fig`` zoom horizontally only, with each y-axis refitted to the visible data in the browser.

    Charts that turn dragging off (``dragmode=False``) and axes the page could not fit (categories, dates on y,
    horizontal bars, histograms, axes already fixed or matched to another) are left as they are.
    """

    layout = fig.layout
    if layout.dragmode is False:
        return fig
    meta = layout.meta
    if meta is not None and not isinstance(meta, dict):
        return fig
    traces_by_axis, unfit = {}, set()
    for trace in fig.data:
        if not hasattr(trace, "yaxis"):
            continue  # a pie, sankey, ...
        key = _axis_key(trace.yaxis, "y")
        horizontal = getattr(trace, "orientation", None) == "h"
        if trace.type not in FIT_TRACE_TYPES or horizontal:
            unfit.add(key)
        traces_by_axis.setdefault(key, []).append(trace)

    specs = {}
    for key, traces in traces_by_axis.items():
        if key in unfit:
            continue
        axis = layout[key]
        if axis.fixedrange or axis.matches or axis.type in ("category", "multicategory", "date"):
            continue
        if axis.overlaying and all(trace.type == "bar" for trace in traces):
            spec = {"volume": True}
        elif axis.range is not None and None not in axis.range:
            spec = {"cap": [float(value) for value in axis.range]}
        else:
            spec = {"fit": True}  # not {}: an empty object is dropped on the way to the browser
        specs[key] = spec
        axis.fixedrange = True
    if specs:
        fig.update_layout(meta={**(meta or {}), "fit_y": specs})
    return fig
