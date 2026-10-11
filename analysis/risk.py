"""Stop, target and position size from support/resistance zones.

Sizing is fixed-fractional: the dollar loss if the stop is hit equals
``account_equity * risk_pct``, scaled by a confidence tier and capped by a
maximum position value.
"""

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RiskSettings:
    account_equity: float = 100_000.0
    risk_pct: float = 1.0
    max_position_pct: float = 25.0
    atr_stop_buffer: float = 0.5
    fallback_stop_atr: float = 2.0
    max_stop_atr: float = 4.0
    min_reward_risk: float = 1.5
    full_size_confidence: float = 75.0
    half_size_confidence: float = 60.0


@dataclass
class TradePlan:
    direction: int
    entry: float
    stop: float
    target: float
    risk_per_share: float
    reward_risk: float
    size_multiplier: float
    shares: int
    position_value: float
    dollar_risk: float
    stop_basis: str
    target_basis: str
    notes: list = field(default_factory=list)
    blocking_zone: object = None

    @property
    def side(self):
        return "LONG" if self.direction > 0 else "SHORT / EXIT"


def size_multiplier_for(confidence, settings):
    """Full size, half size or no trade depending on confidence."""

    if confidence is None or not np.isfinite(confidence):
        return 0.0
    if confidence >= settings.full_size_confidence:
        return 1.0
    if confidence >= settings.half_size_confidence:
        return 0.5
    return 0.0


def plan_trade(direction, entry, atr_value, supports, resistances, settings, confidence=None):
    """Build a trade plan. ``supports``/``resistances`` are nearest-first zones."""

    notes = []
    atr_value = float(atr_value) if atr_value and np.isfinite(atr_value) else entry * 0.02
    buffer = settings.atr_stop_buffer * atr_value

    # Long: stop under the nearest support; target at the nearest resistance.
    # Short: mirror image.
    protective = supports if direction > 0 else resistances
    objective = resistances if direction > 0 else supports

    stop, stop_basis = None, ""
    for zone in protective:
        candidate = zone.low - buffer if direction > 0 else zone.high + buffer
        distance = direction * (entry - candidate)
        if distance < 0.5 * atr_value:
            continue
        if distance > settings.max_stop_atr * atr_value:
            notes.append(
                f"Nearest {'support' if direction > 0 else 'resistance'} zone {zone.describe()} is "
                f"{distance / atr_value:.1f} ATR away - too far, using an ATR stop instead."
            )
            break
        stop = candidate
        stop_basis = f"{settings.atr_stop_buffer:g} ATR beyond {'support' if direction > 0 else 'resistance'} {zone.describe()}"
        break
    if stop is None:
        stop = entry - direction * settings.fallback_stop_atr * atr_value
        stop_basis = f"{settings.fallback_stop_atr:g} ATR from entry (no usable zone)"

    risk_per_share = abs(entry - stop)

    target, target_basis, blocking_zone = None, "", None
    for zone in objective:
        candidate = zone.low if direction > 0 else zone.high
        if direction * (candidate - entry) >= 0.5 * atr_value:
            target = candidate
            target_basis = f"front edge of {'resistance' if direction > 0 else 'support'} {zone.describe()}"
            break
        # Entry sits inside / right against this zone: it has to be cleared first.
        blocking_zone = blocking_zone or zone
    if target is None:
        target = entry + direction * 2.0 * risk_per_share
        if blocking_zone is not None:
            target_basis = f"2R measured move (beyond {blocking_zone.describe()})"
        else:
            target_basis = "2R measured move (no zone beyond entry - price discovery)"
    if blocking_zone is not None:
        notes.append(
            f"Entry {entry:,.2f} is pressed against {'resistance' if direction > 0 else 'support'} "
            f"{blocking_zone.describe()} - it must be cleared before the move has room."
        )

    reward_risk = abs(target - entry) / risk_per_share if risk_per_share > 0 else np.nan
    if np.isfinite(reward_risk) and reward_risk < settings.min_reward_risk:
        notes.append(
            f"Reward/risk to the first target is only {reward_risk:.2f} (< {settings.min_reward_risk:g}); "
            "consider waiting for a pullback toward the trigger level."
        )

    multiplier = size_multiplier_for(confidence, settings)
    risk_budget = settings.account_equity * settings.risk_pct / 100.0 * multiplier
    shares = math.floor(risk_budget / risk_per_share) if risk_per_share > 0 else 0
    max_shares = math.floor(settings.account_equity * settings.max_position_pct / 100.0 / entry) if entry > 0 else 0
    if shares > max_shares:
        notes.append(f"Position capped at {settings.max_position_pct:g}% of equity ({max_shares} shares).")
        shares = max_shares

    return TradePlan(
        direction=direction,
        entry=float(entry),
        stop=float(stop),
        target=float(target),
        risk_per_share=float(risk_per_share),
        reward_risk=float(reward_risk),
        size_multiplier=multiplier,
        shares=int(shares),
        position_value=float(shares * entry),
        dollar_risk=float(shares * risk_per_share),
        stop_basis=stop_basis,
        target_basis=target_basis,
        notes=notes,
        blocking_zone=blocking_zone,
    )
