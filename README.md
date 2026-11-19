# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/abbinaruto/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/abbinaruto/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each GTA VI PlayStation 5 trophy's Exophase earn rate one week after launch, by forecasting earners and players tracked separately and dividing — so a refresh's dilution is never mistaken for a falling rate.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=x4x7sknz&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

## How it works

An earn rate is earners over players tracked, and the two move for
different reasons: players tracked jumps at each Exophase refresh
(new players get counted) while earners grow as the tracked players
progress through the game. Fitting the rate directly confuses the
two, so `main.py` rebuilds both from every read (`earners = rate *
players / 100`) and forecasts each separately:

1. **Segments.** Exophase's page is frozen between refreshes, so
   consecutive identical reads collapse into one state.
2. **Earners.** Per trophy, log(earners) is fit against
   `ln(1 + hours since 2026-11-19T00:00Z)` over the last 36 hours of
   segments, by weighted least squares. Each trophy keeps its own
   slope; the tier's pooled slope is only a fallback for a trophy
   with no fit. Forecasts anchor on the trophy's last observed
   earners.
3. **Players.** Log(players tracked) is fit the same way over the
   same segments.
4. **Divide.** The rate at the target is `100 * earners / players`,
   clamped to [0.01, 99.99]. A trophy at 100% stays at 100%.
5. **Spread.** A random-walk term (`0.5 * sqrt(days)`, the house
   baseline's) plus both fits' prediction-interval widths. Quantiles
   are the median shifted by the normal z-scores on the logit scale.

When the history is a single flat state (launch night, before the
first refresh), there is no slope to fit: the median is the last rate
scaled by the dilution the rules' first refresh implies (638 to
3,868 players tracked), with a wide spread.

## Validation

`validate.py` (kept beside this repo, not shipped) backtests the
forecaster on synthetic histories built from the dynamics the rules
describe — log-linear earner and player growth, Exophase refreshes,
2-decimal rounding — scored by the exact pinball loss, in two
worlds: growth that stays log-linear, and growth that saturates over
the week. Final-like cases, 12 draws each:

| forecaster | log-linear | saturating |
|---|---:|---:|
| house baseline | 0.267 | 0.220 |
| house curve baseline | 0.214 | 0.216 |
| this entry | **0.209** | **0.214** |

Two choices the backtest settled: no slope shrinkage toward the tier
mean (it costs 0.007 in the log-linear world), and a 36-hour fit
window (using the whole history loses badly once growth saturates).
