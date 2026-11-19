"""Forecast GTA VI's PlayStation 5 trophy earn rates on Exophase at the case target.

The idea: an earn rate is earners over players tracked, and the two move
for different reasons. Players tracked jumps at each Exophase refresh (new
players get counted) while earners grow as the tracked players progress
through the game, so the rate itself is a noisy ratio of two moving
quantities. Fitting the rate directly confuses the two. This program
rebuilds both from every read -- earners = rate * players / 100 -- and
forecasts each separately, then divides:

  1. Group the case's reads into segments: Exophase's page is frozen
     between refreshes, so consecutive identical reads are one state.
  2. For each trophy, fit log(earners) against u = ln(1 + hours since
     2026-11-19T00:00Z) over the last 36 hours of segments, by weighted
     least squares (weights ~ earners, since a 2-decimal rate is noisier
     for rare trophies). Each trophy keeps its own fitted slope: the
     backtest showed shrinking slopes toward the tier mean hurts, since
     trophies in one tier climb at genuinely different rates. The tier's
     pooled slope is used only as a fallback for a trophy with no fit of
     its own. Each forecast is anchored on the trophy's last observed
     earners, so a noisy fit cannot drag the level away from the data.
  3. Fit log(players tracked) the same way over the same segments.
  4. Forecast earners and players at the target, divide, and clamp to
     [0.01, 99.99]. A trophy at 100% stays at 100%.
  5. Spread: a random-walk term (0.5 * sqrt(days), the house baseline's)
     plus the two fits' prediction-interval widths, so the band widens
     where the history is thin or the target is far. Quantiles are the
     median shifted by the normal z-scores on the logit scale.

  When the history is a single flat state (launch night, before the
  first refresh), there is no slope to fit. The rules say the first
  refresh took players tracked from 638 to 3,868 and every rate down,
  so the median is the last rate scaled by the dilution that refresh
  implies (about a quarter) with a wide spread, instead of pretending
  the frozen page will hold.

Output: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]},
one row per trophy, exactly as the contest scores it.

Runs offline, standard library only, deterministic (no randomness,
reads in name order, trophies in the order the case lists them), and
fast: a case is 55 trophies over a few dozen reads.

Validated by a synthetic backtest (validate.py beside this repo, not
shipped): histories simulated with the dynamics the rules describe --
log-linear earner and player growth, Exophase refreshes that freeze
the page between them, 2-decimal rounding -- scored by exact pinball
loss against the house baseline and the house curve baseline, in two
worlds: growth that stays log-linear, and growth that saturates over
the week (the realistic one for a launch). The earners-over-players
model beats both baselines in both worlds (final-like cases, 12 draws
each: ours 0.209 / 0.214 vs house curve 0.214 / 0.216 and house
baseline 0.267 / 0.220). Two choices the backtest settled: no slope
shrinkage toward the tier mean (it costs 0.007 in the log-linear
world), and a 36-hour fit window (using the whole history loses badly
once growth saturates, 0.229 vs 0.214).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a
# module shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# The house baseline's random-walk spread, logit units per sqrt day.
SIGMA_RW = 0.5
# Hours of history the trend fits use (the house curve uses 24; a
# little more data steadies the slope without reaching stale regimes).
FIT_WINDOW_HOURS = 36.0
# Shrinkage toward the tier's pooled slope (in segment counts).
# Zero: the backtest showed any shrinkage hurts -- trophies in one
# tier climb at genuinely different rates. The tier pool is only a
# fallback for a trophy with no fit of its own.
SHRINK_K = 0.0
# A flat launch-night history's median is scaled by this, the dilution
# the rules' first refresh implies (638 -> 3,868 players tracked).
FLAT_DILUTION = 0.25
# Spread when no fit exists (flat history): the refresh is coming.
FLAT_SIGMA = 1.0
SIGMA_FLOOR = 0.4
SIGMA_CAP = 2.5
# The house curve's clock: hours from 2026-11-19T00:00:00Z.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, like JavaScript's Math.round."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def u_of(when):
    """ln(1 + hours since the launch-night epoch), the trend's clock."""
    hours = (when - EPOCH).total_seconds() / 3600.0
    return math.log(max(0.05, 1.0 + hours))


def segments(case):
    """One observation per Exophase state: consecutive identical reads
    are the same frozen page. Returns a list of (u, at, players, rates)
    oldest first, each state once, at its first read's time."""
    out = []
    prev_key = None
    for read in case.get("history", []):
        rates = read.get("rates") or []
        key = (read.get("players_tracked"),
               tuple(-1 if r is None else r for r in rates))
        if key == prev_key:
            continue
        prev_key = key
        out.append((u_of(parse_ts(read["at"])), parse_ts(read["at"]),
                    read.get("players_tracked"), rates))
    return out


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns
    (intercept, slope, resid_sd, n, xbar, sxx) or None when the fit
    is degenerate (fewer than 2 distinct x)."""
    n = len(xs)
    if n < 2:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    dof = max(n - 2, 1)
    resid = math.sqrt(sum(w * (y - (intercept + slope * x)) ** 2
                          for x, y, w in zip(xs, ys, ws)) / (dof * (sw / n)))
    return (intercept, slope, resid, n, xbar, sxx)


def pred_sd(fit, u_target):
    """Prediction-interval width (in the fit's units) at u_target."""
    if fit is None:
        return None
    _, _, resid, n, xbar, sxx = fit
    lev = 1.0 + 1.0 / n + (u_target - xbar) ** 2 / sxx
    return resid * math.sqrt(max(lev, 1.0))


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    segs = segments(case)
    if not segs:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)
    at = parse_ts(case["at"])
    u_target = u_of(parse_ts(case["target"]))
    days = case["horizon_minutes"] / 1440.0
    rw = SIGMA_RW * math.sqrt(days)

    # Trend window: the last 36 hours of states; all states when fewer
    # than 3 fall in it (a flat launch night has exactly one state).
    cutoff = at.timestamp() - FIT_WINDOW_HOURS * 3600.0
    window = [s for s in segs if s[1].timestamp() >= cutoff]
    if len(window) < 3:
        window = segs
    w_us = [s[0] for s in window]

    # --- players tracked: log N against u, anchored on the last state.
    log_n = [math.log(max(1, s[2])) for s in window]
    n_fit = wls(w_us, log_n, [1.0] * len(window))
    n_sd = pred_sd(n_fit, u_target)
    last_n = segs[-1][2]
    log_n_last = math.log(max(1, last_n))
    if n_fit is not None:
        d_n = n_fit[1]
        log_n_target = log_n_last + d_n * (u_target - segs[-1][0])
    else:
        log_n_target = log_n_last

    # --- per-trophy earner fits: log E against u, E = rate * N / 100.
    fits = []
    for i in range(n_tr):
        xs, ys, ws = [], [], []
        for s in window:
            rate = s[3][i] if i < len(s[3]) else None
            if rate is None:
                continue
            n_players = max(1, s[2])
            earners = rate / 100.0 * n_players
            if rate >= 100:
                earners = float(n_players)
            earners = max(earners, 0.5)
            xs.append(s[0])
            ys.append(math.log(earners))
            ws.append(earners)
        fits.append(wls(xs, ys, ws))

    # Tier-pooled slopes: trophies with a real fit vote, weighted by
    # how many states their fit used; the pool steadies rare trophies.
    tier_slopes = {}
    all_slopes = []
    for i, tr in enumerate(trophies):
        f = fits[i]
        if f is not None and f[3] >= 3:
            tier_slopes.setdefault(tr["tier"], []).append((f[1], f[3]))
            all_slopes.append((f[1], f[3]))
    def pooled(pairs):
        if not pairs:
            return None
        sw = sum(w for _, w in pairs)
        return sum(s * w for s, w in pairs) / sw
    overall_slope = pooled(all_slopes)
    tier_pool = {t: pooled(p) for t, p in tier_slopes.items()}

    # Last observed earners per trophy (most recent state naming it).
    last_e = []
    last_u = []
    for i in range(n_tr):
        e, u = None, None
        for s in reversed(segs):
            rate = s[3][i] if i < len(s[3]) else None
            if rate is None:
                continue
            n_players = max(1, s[2])
            e = float(n_players) if rate >= 100 else max(rate / 100.0 * n_players, 0.5)
            u = s[0]
            break
        last_e.append(e)
        last_u.append(u)

    flat = len(segs) == 1
    rows = []
    for i, tr in enumerate(trophies):
        last_rate = segs[-1][3][i] if i < len(segs[-1][3]) else None
        if last_rate is None:
            last_rate = 0.0
        if last_rate >= 99.99:
            # Everybody tracked already has it; it stays earned.
            rows.append({"id": tr["id"], "q": [100, 100, 100, 100, 100]})
            continue
        f = fits[i]
        if flat:
            # No slope exists yet: the next refresh dilutes the rate.
            median = clamp(last_rate * FLAT_DILUTION)
            sigma = math.sqrt(rw * rw + FLAT_SIGMA * FLAT_SIGMA)
        else:
            pool = tier_pool.get(tr["tier"], overall_slope)
            if f is not None:
                n_i = f[3]
                slope = (n_i * f[1] + SHRINK_K * (pool if pool is not None else 0.0)) / (n_i + SHRINK_K)
                e_sd = pred_sd(f, u_target)
            else:
                slope = pool if pool is not None else 0.0
                e_sd = FLAT_SIGMA
            log_e_target = math.log(last_e[i]) + slope * (u_target - last_u[i])
            median = clamp(100.0 * math.exp(log_e_target - log_n_target))
            sigma = math.sqrt(rw * rw + (e_sd or 0.0) ** 2 + (n_sd or 0.0) ** 2)
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
        lg = logit(median)
        qs = []
        prev = 0.0
        for z in Z:
            q = round4(100.0 * sigmoid(lg + z * sigma))
            q = min(max(q, 0.01), 100)
            q = max(q, prev)
            qs.append(q)
            prev = q
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
