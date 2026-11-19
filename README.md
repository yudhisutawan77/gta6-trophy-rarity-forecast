# GTA VI Trophy Rarity Forecast

![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)
![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yudhisutawan77/gta6-trophy-rarity-forecast/rank.svg)
![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yudhisutawan77/gta6-trophy-rarity-forecast/reward.svg)

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on [Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26T00:00:00Z — a week after launch — as five quantiles per trophy.
Built for the make.fun bounty
[Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
($250, 20 paid places).

## The model

Each trophy is modelled in logit space, `logit(p) = ln(p / (1 - p))`, against
`x = ln(1 + hours since release)`, because adoption curves are roughly linear
in log-time: early story trophies climb fast then saturate, late-game trophies
climb slowly. Over the last 36 hours of reads a least-squares line is fitted
per trophy; its slope is the *net* drift — earner growth minus dilution from
new players joining Exophase tracking — shrunk toward zero by its own
t-statistic and by the number of distinct reads, then extrapolated to the
target (move capped at 3.5 logit). The spread combines a random-walk term
growing with the horizon, the slope's standard error over the extrapolation
distance, the fit's residual scale, and a refresh-jump term (Exophase
refreshes its counts on its own schedule, so the target read can sit just
after a refresh the checkpoint read has not seen).

Two cases need special handling. When the history is flat — every read
identical, as on launch night — there are no dynamics to fit, so the median
is shifted down 0.65 logit (a refresh dilutes every non-universal rate: the
tracked-player count jumps while earners barely move) and the spread is set
wide (1.1 logit) so the 90% band can still cover a several-fold dilution.
The one trophy at 100% is forecast to stay ~100% with a narrow spread: it
survived a six-fold jump in tracked players unchanged.

## Running it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in` in the
input directory and writes one `<id>.out` per case:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.
Fully deterministic — two runs produce byte-identical output — and well
inside the 8-second limit per case.

## License

MIT — see [LICENSE](LICENSE).
