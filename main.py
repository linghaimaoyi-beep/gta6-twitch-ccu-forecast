"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

The program reads one case file (JSON, schema makefun/twitch-ccu-case/v1):
hour 1's print for the launch, the hours 2..24 to forecast, and the
comparables: past big game launches with their full hourly viewer curves.
It writes one forecast file: for each of hours 2..24, five quantiles of that
hour's concurrent viewers in the game's Twitch category.

Model
-----
Launch days share a shape once you anchor on hour 1, but the shape depends
strongly on how big hour 1 is: a launch whose first hour drew a few hundred
viewers can peak a thousand times higher by evening, while a launch whose
first hour already drew close to a million peaks at barely its first-hour
level (across the comparables, log hour-1 viewers and the peak log-ratio
correlate at about -0.94). So the forecast is conditioned on the target's
own hour-1 scale.

For each hour h, every comparable contributes its log-ratio
r = ln(viewers_h / viewers_1) together with its log hour-1 level. The
centre of r_h at the target's level is a locally weighted linear fit of r_h
on log hour-1 (Gaussian kernel, bandwidth 1.0 in log units), clipped to the
range the eight nearest comparables actually show, so a launch bigger than
any comparable cannot be extrapolated off a cliff. The spread is the
kernel-weighted quantiles (bandwidth 1.5) of the residuals of each
comparable from its own local fit, which keeps the width honest for the
target's scale: big launches are tight, small ones are wide. The five
quantile curves are lightly smoothed across neighbouring hours, because the
true shape moves smoothly and a single hour's handful of comparables is
noisy. Quantiles are built on the log scale because the score is a pinball
loss on ln(viewers), so an honest spread is what scores.

Run as: python3 main.py  (reads $MAKEFUN_INPUT_DIR or /input, writes
$MAKEFUN_OUTPUT_DIR or /output). Standard library only, no network,
deterministic: the same case always writes the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
# Bandwidths (in natural-log units of hour-1 viewers) for the local fit of
# the centre and for the kernel-weighted residual quantiles.
BANDWIDTH_CENTRE = 1.0
BANDWIDTH_SPREAD = 1.5
# Fallback spread (the house baseline's), used only when a case carries no
# usable comparable for an hour.
FALLBACK_SIGMA = 0.15


def _parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _round2(x):
    """Round to 2 decimals, half up, as a plain number (ints stay ints)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def _release_day_ratios(case):
    """Every comparable that has a release day in its curve, as
    (log hour-1 viewers, {hour: log-ratio to hour 1 for hours 2..24}).
    A comparable needs hour 1 and at least 15 of the 23 later hours, so a
    sparse or truncated curve cannot steer the fit.
    """
    out = []
    for comp in case.get("comparables") or []:
        hourly = comp.get("hourly") or []
        if not hourly:
            continue
        start = _parse_ts(comp["day_start"])
        by_hour = {}
        for point in hourly:
            viewers = point.get("viewers")
            if viewers is None or viewers <= 0:
                continue
            delta = (_parse_ts(point["from"]) - start).total_seconds() / 3600.0
            if 0 <= delta < 24 and delta == int(delta):
                by_hour[int(delta)] = float(viewers)
        first = by_hour.get(0)
        if not first or len(by_hour) < 16:
            continue
        log_first = math.log(first)
        ratios = {}
        for h in range(2, 25):
            value = by_hour.get(h - 1)
            if value:
                ratios[h] = math.log(value) - log_first
        if ratios:
            out.append((log_first, ratios))
    return out


def _kernel_weight(level, target, bandwidth):
    return math.exp(-0.5 * ((level - target) / bandwidth) ** 2)


def _weighted_linreg(xs, ys, weights):
    """Weighted least squares of y on x; returns (intercept, slope)."""
    total = sum(weights)
    if total <= 0:
        return 0.0, 0.0
    mean_x = sum(w * x for w, x in zip(weights, xs)) / total
    mean_y = sum(w * y for w, y in zip(weights, ys)) / total
    sxx = sum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs))
    sxy = sum(w * (x - mean_x) * (y - mean_y)
              for w, x, y in zip(weights, xs, ys))
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    return mean_y - slope * mean_x, slope


def _weighted_quantile(pairs, level):
    """Weighted quantile of (value, weight) pairs, by cumulative weight."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return None
    target = level * total
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= target:
            return value
    return ordered[-1][0]


def _hour_quantiles(data, log_p1, hour):
    """The five log-ratio quantiles for one hour, or None when no comparable
    covers it."""
    pairs = [(level, ratios[hour]) for level, ratios in data if hour in ratios]
    if not pairs:
        return None
    centre_weights = [_kernel_weight(level, log_p1, BANDWIDTH_CENTRE)
                      for level, _ in pairs]
    intercept, slope = _weighted_linreg(
        [level for level, _ in pairs],
        [value for _, value in pairs],
        centre_weights,
    )
    centre = intercept + slope * log_p1
    # Never let the fit run past what launches near this size actually did.
    nearest = sorted(pairs, key=lambda p: abs(p[0] - log_p1))[:8]
    low = min(value for _, value in nearest)
    high = max(value for _, value in nearest)
    centre = max(low, min(high, centre))
    # Residuals of each comparable from its own local fit, weighted by how
    # close its size is to the target's: the spread for this target's scale.
    residuals = []
    for level, value in pairs:
        weight = _kernel_weight(level, log_p1, BANDWIDTH_SPREAD)
        residuals.append((value - (intercept + slope * level), weight))
    return [centre + _weighted_quantile(residuals, level) for level in LEVELS]


def _smoothed_quantiles(data, log_p1):
    """Per-hour, per-level log-ratio quantiles, smoothed across neighbouring
    hours so the shape cannot jump on one hour's noisy handful."""
    raw = {h: _hour_quantiles(data, log_p1, h) for h in range(2, 25)}
    smooth = {}
    for h in range(2, 25):
        neighbours = [hh for hh in (h - 1, h, h + 1) if 2 <= hh <= 24]
        smooth[h] = []
        for i in range(len(LEVELS)):
            usable = [raw[hh][i] for hh in neighbours
                      if raw[hh] is not None and raw[hh][i] is not None]
            smooth[h].append(sum(usable) / len(usable) if usable else None)
    return smooth


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the five contest levels")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)

    data = _release_day_ratios(case)
    quantiles = _smoothed_quantiles(data, log_p1)

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = quantiles.get(h) or [None] * len(LEVELS)
        row_q = []
        for i, level in enumerate(LEVELS):
            value = qs[i]
            if value is None:
                # No comparable covers this hour: fall back to persistence
                # with a widening spread.
                value = Z[i] * FALLBACK_SIGMA * math.sqrt(h - 1)
            row_q.append(_round2(math.exp(log_p1 + value)))
        # Keep the five quantiles non-decreasing whatever the data did.
        for i in range(1, len(row_q)):
            if row_q[i] < row_q[i - 1]:
                row_q[i] = row_q[i - 1]
        row_q = [max(1, min(1_000_000_000, q)) for q in row_q]
        hours.append({"hour": h, "q": row_q})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
