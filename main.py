"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn
rate at the case target, as five quantiles on the logit scale.

Model (per trophy, all in logit space, logit(p) = ln(p / (1 - p)) with the
rate clamped to 0.01..99.99 first):

1.  Time is measured as x = ln(1 + hours since release, 2026-11-19T00:00Z).
    Adoption curves are roughly linear in ln-time: early story trophies
    climb fast then saturate, late-game trophies climb slowly.
2.  Over the last 36 h of reads (or all reads when fewer exist) fit, by least
    squares, logit(rate) = a + b * x.  The slope b is the *net* drift: earner
    growth minus dilution from new players joining Exophase tracking, so no
    separate dilution term is needed.  The slope is shrunk toward zero by
    its own t-statistic (a noisy slope is mostly noise) and by the number of
    distinct reads, and the extrapolated move is capped at 3.5 logit.
3.  Spread (logit sd) combines a random-walk term growing with the horizon,
    the slope's standard error times the extrapolation distance, the fit's
    residual scale, and a small refresh-jump term: Exophase refreshes its
    counts on its own schedule, so the target read can sit just after a
    refresh the checkpoint read has not seen.
4.  Flat history (every read identical, as on launch night) carries no
    dynamics at all: the median is shifted down 0.65 logit -- a refresh
    dilutes every non-universal rate, because the tracked-player count
    jumps while earners barely move -- and the spread is set wide (1.1
    logit) so the 90% band can still cover a several-fold dilution.
5.  The 100% trophy (earned by every tracked player, including newcomers)
    is forecast to stay ~100% with a narrow spread; a trophy with no usable
    history falls back to persistence with a wide spread.

Everything is deterministic: standard library only, no network, no clock,
no randomness.  Reads every *.in in $MAKEFUN_INPUT_DIR (default /input),
writes /output/<id>.out (default $MAKEFUN_OUTPUT_DIR).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Random-walk volatility of a logit's move, per sqrt(day).
RW_SIGMA = 0.45
# Logit sd of an unseen refresh jump (target read refreshed, checkpoint not).
JUMP_SIGMA = 0.25
# Flat-history mode: median shift (dilution) and spread, in logit units.
FLAT_SHIFT = -0.65
FLAT_SIGMA = 1.10
# How far a trend extrapolation may move the median, in logit units.  Wide
# on purpose: an early story trophy can climb from ~1% toward ~30% over the
# final's 4.5 days, a move of ~3.7 logit; the slope shrinkage, not this cap,
# is what keeps a noisy fit near persistence.
MAX_MOVE = 3.5
# Window (hours) of reads the trend fit uses when enough exist.
FIT_WINDOW_HOURS = 36.0


def parse_time(text):
    """ISO 8601 -> aware datetime; 'Z' and date-only forms accepted."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def sigmoid(lg):
    lg = max(-30.0, min(30.0, lg))
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def hours_since_release(dt):
    return max(0.0, (dt - RELEASE).total_seconds() / 3600.0)


def ln_time(dt):
    return math.log1p(hours_since_release(dt))


def ols(xs, ys):
    """Least squares y ~ a + b x. Returns (a, b, resid_sd, slope_se, n)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return my, 0.0, 0.0, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        var = resid / (n - 2)
    else:
        var = 0.0
    resid_sd = math.sqrt(var)
    slope_se = math.sqrt(var / sxx) if var > 0.0 else 0.0
    return a, b, resid_sd, slope_se, n


def dedup_reads(history):
    """Drop consecutive reads that repeat players and rates exactly: a stale
    Exophase page re-read many times must not count as many observations."""
    kept = []
    for read in history:
        if kept and read.get("players_tracked") == kept[-1].get("players_tracked") \
                and read.get("rates") == kept[-1].get("rates"):
            continue
        kept.append(read)
    return kept


def forecast_trophy(idx, case, history, x_target, horizon_days):
    """Return the five quantiles for trophies[idx]."""
    trophy = case["trophies"][idx]
    tier = trophy.get("tier", "bronze")

    # (x, logit rate) points for this trophy, oldest first.
    points = []
    last_rate = None
    last_x = None
    for read in history:
        rates = read.get("rates") or []
        if idx >= len(rates):
            continue
        rate = rates[idx]
        if rate is None:
            continue
        x = ln_time(parse_time(read["at"]))
        points.append((x, logit(rate)))
        last_rate = rate
        last_x = x

    rw = RW_SIGMA * math.sqrt(max(horizon_days, 0.0))

    if last_rate is None:
        # Trophy never seen in any read: tier prior, very wide spread.
        prior = {"platinum": 0.3, "gold": 0.6, "silver": 0.8}.get(tier, 1.0)
        return quantiles(logit(prior), 1.5)

    if last_rate >= 99.99:
        # Universal trophy: every tracked player has it, newcomers included.
        return quantiles(logit(99.99), 0.5)

    # Flat history: no observed movement at all.
    distinct = {(round(x, 9), round(y, 9)) for x, y in points}
    if len(distinct) <= 1 or len(points) < 2:
        spread = math.sqrt(FLAT_SIGMA ** 2 + rw ** 2)
        return quantiles(logit(last_rate) + FLAT_SHIFT, spread)

    # Trend fit over the recent window when it holds enough distinct reads.
    # The window is the last FIT_WINDOW_HOURS of wall-clock time, expressed
    # in ln-time: x >= ln(1 + hours_last - 36).
    hours_last = math.expm1(last_x)
    cutoff_x = math.log1p(max(0.0, hours_last - FIT_WINDOW_HOURS))
    recent = [(x, y) for x, y in points if x >= cutoff_x]
    fit_points = points
    if len({round(x, 6) for x, _ in recent}) >= 3:
        fit_points = recent
    xs = [x for x, _ in fit_points]
    ys = [y for _, y in fit_points]
    _a, slope, resid_sd, slope_se, n = ols(xs, ys)

    # Shrink the slope: by its t-statistic (noisy slope -> ~0) and by the
    # number of distinct reads (few reads -> mostly persistence).
    distinct_x = len({round(x, 6) for x in xs})
    if slope_se > 0.0 and slope != 0.0:
        t = abs(slope) / slope_se
        t_shrink = (t * t) / (t * t + 1.0)
    else:
        t_shrink = 1.0 if slope != 0.0 else 0.0
    n_shrink = distinct_x / (distinct_x + 3.0)
    slope_s = slope * t_shrink * n_shrink

    dx = x_target - last_x
    move = max(-MAX_MOVE, min(MAX_MOVE, slope_s * dx))
    median_logit = logit(last_rate) + move

    # Spread: random walk + slope uncertainty over the extrapolation +
    # residual scale + refresh jump.  Floored so a perfect-looking fit
    # never claims a near-certain forecast over days.
    spread = math.sqrt(rw ** 2 + (slope_se * dx) ** 2
                       + resid_sd ** 2 + JUMP_SIGMA ** 2)
    spread = max(spread, 0.35)
    spread = min(spread, 2.2)
    return quantiles(median_logit, spread)


def quantiles(median_logit, spread):
    out = []
    prev = None
    for z in Z:
        q = round4(sigmoid(median_logit + z * spread))
        q = min(100.0, max(0.0, q))
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}")
    history = dedup_reads(case.get("history") or [])
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    target = case.get("target")
    if target is None:
        target = case["at"]
    x_target = ln_time(parse_time(target))
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(i, case, history, x_target,
                                          horizon_days)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
