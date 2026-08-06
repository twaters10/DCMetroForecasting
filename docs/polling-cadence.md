# Polling cadence: what a finer interval would buy, and why the cheap route fails

**Current state: 60 seconds.** A move to 30s was attempted and reverted. This documents
what the change is worth, the two ways to deploy it, and the measured reason the free
one does not work — so nobody repeats the attempt from first principles.

## Why a finer cadence is worth something

Arrivals are derived, not reported. The feeds carry state, not events, so a
`current_stop_sequence` increment between two snapshots means the train arrived
somewhere in that gap — **the gap is the error bar.**

Measured on real rail data at 60s:

| | 60s | 30s |
| --- | --- | --- |
| Arrival error | ±30s | ±15s |
| Relative error on the median 120s segment | ~25% | ~12.5% |
| Stop traversals passed unobserved | 8.9% | expected to fall materially |

At 60s, `delay_sec` reads p05 −60 / median 0 / p95 +60 — those are polling bins, not
delays. Per-row delay is not a measurement, only aggregates are.

## EventBridge Scheduler cannot go below one minute

Both forms are rejected outright. Verified against the live API:

```
$ aws scheduler create-schedule --schedule-expression 'rate(30 seconds)' ...
ValidationException: Invalid Schedule Expression rate(30 seconds).

$ aws scheduler create-schedule --schedule-expression 'cron(*/30 * * * * ? *)' ...
ValidationException: Invalid Schedule Expression cron(*/30 * * * * ? *).
```

So a sub-minute archive needs either two schedules, or two passes inside one invocation.

## The two-schedule approach does not work — measured

The idea: two `rate(1 minutes)` schedules whose `StartDate` values sit 30s apart, giving
firings at `:00` and `:30` and costing nothing, since each invocation stays ~2s and the
whole thing fits in the Lambda free tier.

**It fails, because EventBridge Scheduler discards the sub-minute component of
`StartDate` for rate-based schedules.** It aligns firings to the minute boundary; the
seconds you supply are ignored, and whatever offset you observe is that schedule's own
delivery latency.

Tried it. The second schedule was created with `StartDate` `18:59:30Z` and
`FlexibleTimeWindow: OFF`. AWS retained the value — `get-schedule` still reports
`2026-08-06T14:59:30-04:00` — and then fired at `:10` every minute regardless:

```
19:00:00  19:00:10   19:01:00  19:01:10   19:02:00  19:02:10   19:03:00  19:03:10
gaps: [60, 10, 50, 10, 50, 10, 50, 10, 50, 10, 50]
```

Stable, and stably wrong. That is not a 30s cadence — it is 120 snapshots/hour with
brackets alternating 10s and 50s, so half the arrivals land in a bracket barely better
than the 60s baseline. Deleting and recreating might land nearer `:30` by luck, but the
latency is not a knob we control and could shift at any time.

**Do not retry this approach.** It looks free and correct on paper and is neither.

> Watch out for a bad test here. The first check declared "30s cadence confirmed"
> because it took the *median* of the gaps — and `[10,10,10,10,10,50,50,50,50,60]` has a
> median of exactly 30. A bimodal distribution can pass a median test while being the
> opposite of even. Check that **every** gap is 30, not that the middle one is.

## The approach that does work, and its cost

One schedule at `rate(1 minutes)` plus `POLLS_PER_INVOCATION=2` on the Lambda. The
handler polls, waits to an absolute target 30s later, and polls again. Spacing is exact
and owes nothing to EventBridge latency.

The catch is that **Lambda bills the wait**:

| | Invocations/day | Duration each | Lambda cost/month |
| --- | --- | --- | --- |
| current, 60s | 1,440 | ~2s | $0.00 (inside free tier) |
| `POLLS_PER_INVOCATION=2` | 1,440 | ~34s | **~$11.40** |

At 1024 MB arm64, against a 400,000 GB-s free tier. S3 adds ~172,800 PUTs/month ≈ $0.86
and double the stored bytes under the 90-day expiry.

To enable it, set one environment variable on the function:

```
POLLS_PER_INVOCATION=2
```

and confirm `Timeout` is ≥180s (already 300s). Nothing else changes. Verified locally:
two passes wrote snapshots exactly 30s apart with distinct keys.

## The code is ready either way

`POLLS_PER_INVOCATION` defaults to 1, so the collector behaves exactly as it always has
until the variable is set. Two details in `handler._collect_realtime` matter if it ever is:

- **Passes sleep to an absolute target, not for a fixed duration.** `sleep(30)` after a
  pass that took 8s puts the next one 38s later and the error compounds; snapshots would
  drift within the minute instead of landing near `:00` and `:30`.
- **Each pass takes a fresh `captured_at`, guaranteed to be in a later whole second.**
  Object keys are stamped with `int(captured_at.timestamp())`, so two passes inside one
  second would build the same key and the second write would silently overwrite the
  first — paying for a poll that archives nothing.

## The ETL never assumed 60s, and still doesn't

Nothing in the pipeline needs changing if the cadence moves, because the derivation
always measured the gap rather than reading a constant:

```python
bracket = F.unix_timestamp("captured_at") - F.unix_timestamp("prev_captured_at")
```

Three supporting pieces were made cadence-aware while investigating, and all are worth
keeping regardless:

**Coverage detection is measured, not assumed.** `EXPECTED_SNAPSHOTS_PER_HOUR` is now a
fallback; the real figure comes from `archive.modal_interval_seconds` per window. Had it
stayed hardcoded at 60, a 30s archive would have made every hour trivially clear a
54-file bar — *including an hour holding 60, which is half the data missing* — and the
pipeline's only collector-downtime detector would have silently stopped detecting.
The mode, not the mean, so one outage gap cannot skew it.

**The negative-duration bound uses each row's own bracket.** A −45s duration is a
plausible rounding artefact within a 60s bracket and impossible within a 30s one, so a
fixed constant is wrong for one era or the other.

**`MAX_ARRIVAL_BRACKET_SEC` stays absolute at 180s, deliberately.** What a consumer cares
about is "how wrong can this arrival be", and 180s of uncertainty is 180s however many
polls it spans. A finer interval does not move the bar; more rows simply clear it,
which is correct because they really are more precise.

## If the cadence does change

Old data does not become useless. Every row carries `arrival_bracket_sec`, measured from
the two snapshots that bracketed that specific arrival, so eras are distinguishable and
filterable and nothing needs reprocessing.

For modelling this matters: training across a cadence boundary means heteroscedastic
noise, with earlier rows genuinely noisier. Use `arrival_bracket_sec` as a feature or a
sample weight rather than ignoring it.
