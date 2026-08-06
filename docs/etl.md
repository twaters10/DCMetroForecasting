# Stage 2 — the ETL: how an arrival is derived, and what that costs

The archive holds **state, sampled at a fixed interval** — currently 60 seconds (see
[`polling-cadence.md`](polling-cadence.md) for what a finer cadence would buy). It contains no record saying
"vehicle arrived at stop X at time T". Every arrival in the segment table is derived,
and this document is about how, why that way, and what the derivation cannot tell you.

Every figure below is measured on real archived data — a 3-hour rail window
(2026-08-05 11:00–14:00 UTC, 360 snapshots, 511 trips) unless stated. None of it is
assumed from the GTFS-realtime specification, because on two important points WMATA
does not follow it.

## The two candidate sources, and what the data said

### VehiclePositions — chosen as primary

When a vehicle's `current_stop_sequence` increments between consecutive snapshots, it
arrived at the new stop somewhere in that interval. The arrival is **bracketed** by the
two capture times: the estimate is the midpoint and the bracket width is a genuine
error bar, not an assumption.

| | |
| --- | --- |
| arrival transitions found | 6,119 |
| bracket ≤ 180s (usable) | 98.7% |
| median bracket | 60s |
| advancing exactly one stop | 98.8% |
| (trip, stop) pairs ever seen `STOPPED_AT` | 91.4% |

**That the increment marks arrival and not departure was verified, not assumed** — the
two differ by an entire segment, so getting it backwards would shift every measurement
by one stop. At the newly-incremented sequence the reported status is `STOPPED_AT`
92.3% of the time, and 98.9% of (trip, stop) pairs are already `STOPPED_AT` at first
sighting.

A consequence worth knowing: **WMATA uses `IN_TRANSIT_TO` with sequence N to mean
"departed N"**, not the specification's "heading to N". It is the last status seen at a
sequence 54.1% of the time. Reading it the spec's way would shift every arrival by one
segment.

### TripUpdates — fallback, but not by the obvious method

The natural estimator is "the last prediction before the stop disappears from the
feed". **That estimator does not exist here.** WMATA never removes a served stop:

| | |
| --- | --- |
| trips whose minimum `stop_sequence` ever advances | **0 of 511** |
| trips whose stop list shrinks at all | 2% (and those are truncations at the far end) |
| rows predicting an arrival already in the past | 28.6% |
| passed stops whose prediction then freezes | 7.5% |

So the feed keeps re-predicting a stop forever, and the value **keeps drifting later**
after the train has gone — a further median +53s. Taking the final observed value
therefore reads a stale drift rather than an arrival: it lands a median **+58s** after
the observed truth, with only 53.7% of estimates within 60s.

The estimator that does work is the **last prediction issued while the stop was still
in the future**, for stops observed crossing their own predicted arrival. Measured
against VehiclePositions' observed arrivals over 6,036 overlapping pairs:

| estimator | median error | p10 | p90 | within 60s |
| --- | --- | --- | --- | --- |
| final (drifted) value | **+58s** | +28s | +108s | 53.7% |
| **last forward-looking value** | **+2s** | −25s | +49s | **91.3%** |

No bias correction is applied, because none is needed once the right value is read. An
earlier version of this pipeline subtracted a 58-second constant; that was treating an
artifact of a bad estimator as a property of WMATA's predictor.

### How they combine

VehiclePositions wins wherever it fired. TripUpdates fills the gaps — the ~8.9% of
stop traversals passed between polls. Every row records `arrival_source`,
`arrival_bracket_sec`, and `arrival_confident`, so downstream modelling can weight or
filter on provenance rather than trusting a blend. On a full service day the split is
roughly 95% observed, 5% predicted.

## The sampling-quantization limitation

**Arrival timestamps carry up to half the polling interval in quantization error, and
the segment durations built from them up to a full interval.** At the current 60s
cadence that is ±30s and ±60s; halving the interval would halve both. This is a direct consequence of the
polling rate and cannot be reduced without polling faster — see
[`polling-cadence.md`](polling-cadence.md).

**Never assume the cadence** even so. Every row carries `arrival_bracket_sec`, measured
from the two snapshots that actually bracketed that arrival, and the pipeline measures the
interval per window rather than reading a constant. If the cadence ever changes, the
archive spans both eras and that column is what keeps the two comparable — use it as a
feature or a sample weight rather than assuming a single number.

What that means in practice: the median rail segment is 120 seconds. A ±60s error on a
120s measurement is large in relative terms, and it shows in the 60s-era output —
`delay_sec` has p05 −60, median 0, p95 +60. **Those are quantization bins, not delay
measurements.** At 30s the bins halve and real per-segment variation starts to resolve.
Either way, an individual row is not precise enough to say "this train lost 45 seconds
on this segment".

The dataset is still useful, because the error is close to unbiased (the midpoint
estimator is unbiased if arrival is uniform within the interval) and averages out over
many observations. Aggregate statements — "this segment runs 40 seconds slower at 08:00
than at 14:00" — are well supported once enough traversals accumulate. Per-row
statements are not.

Two smaller precision notes:

- `vehicle.timestamp` is populated on 100% of records and is only **median 7s, max 12s**
  stale. It is not currently used as the arrival anchor; doing so could tighten the
  bracket and is the most promising available accuracy improvement.
- Rows where the vehicle dropped out of the feed and reappeared carry a wide bracket
  (4.9% of segments on a full day). They are flagged `arrival_confident = false` rather
  than dropped, so filtering is the consumer's choice.

## Dwell time is inside the duration

`actual_departure_ts` is **the upstream arrival, not a departure.** WMATA populates
`departure.time` on only 4.5% of stop_time_updates, exclusively at `stop_sequence` 1,
and never alongside an arrival on the same stop (`both` = 0.0%). A true intermediate
departure simply is not published.

So `actual_duration_sec` includes the dwell at the upstream stop. For a 120-second rail
segment a 20–30 second dwell is a meaningful share, so this is a real limitation rather
than a rounding detail. `scheduled_duration_sec` is computed the same way — upstream
scheduled arrival to downstream scheduled arrival — so the two sides are comparable and
`delay_sec` is not contaminated by the asymmetry.

## The static-GTFS versioning concern

Fully documented in [`static-gtfs.md`](static-gtfs.md); the parts that constrain the
ETL:

**The join is two hops on `scheduled_trip_id`.** Realtime `trip_id` is
`{scheduled_trip_id}_{schedule_version}`, and so is static `trips.trip_id`, but the
versions differ whenever WMATA has published a new timetable. Joining them directly
matches **0%**; going via `trips.scheduled_trip_id` and then to `stop_times` on the
versioned id matches **100%**. `stop_times` is keyed on the versioned id, so `trips.txt`
is a required hop rather than a convenience.

**Realtime lags static.** The bundle published on 2026-08-05 declares
`feed_start_date 20260805`, but realtime snapshots archived at 08:00 EDT *on Aug 5*
still emitted `_20660` trip_ids. So which timetable was in force cannot be resolved from
the service date alone. Every output row therefore carries both `static_gtfs_version`
and `schedule_version_agrees`; when they disagree the join is still valid but the
scheduled times may not be the ones that were in force, and `delay_sec` on those rows is
lower confidence.

**Bundles expire.** Rail's feed window is about ten days and WMATA serves only the
current bundle, so the ETL reads from the collector's `static/` archive in S3, never
live from WMATA. Fetching live would silently join historical data against a newer
timetable.

**`route_id == "NR"` must be excluded before computing a match rate.** Non-revenue
equipment moves have no schedule by definition; counting them understates the rate and
makes a real regression indistinguishable from normal operations.

## A trip_id is not a journey

WMATA reuses a `trip_id` within a service day. Trip `12072038_20660` traverses its whole
stop sequence twice inside one 3-hour window. Keying on `trip_id` alone paired stop 11 of
the second run with stop 10 of the first and produced durations of **−4,800 seconds**.

Arrivals are therefore sessionised into `trip_run`: ordered by time, a new run starts
wherever `stop_sequence` fails to advance. On a full service day **17.9% of segment rows
belong to a repeat run.**

This exposes a limitation it does not solve: static GTFS holds one scheduled timetable
per `trip_id`, so a second run has no schedule of its own and its `delay_sec` is measured
against the first run's times. `trip_run > 0` identifies those rows.

## Why decoding happens outside Spark

Protobuf decoding is a pure-Python parse; it cannot vectorise, so a Spark UDF over
`binaryFile` moves the same per-record work onto executors and adds serialisation.

The decisive factor is object shape. A month is ~43k objects per feed at 6 KB–1 MB
each. `binaryFile` assigns roughly one task per file, so that is 43k tasks whose
scheduling overhead exceeds the work in each — and the job is almost entirely S3
round-trip latency, which a 16-thread pool saturates far better on one machine than a
few local executors doing blocking reads.

So stage A is a threaded Python decode writing Parquet, and stage B is Spark. Three
concrete wins: the small-file problem is solved before Spark sees the data; derivation
becomes re-runnable without re-downloading (`--skip-decode`, which matters because this
logic gets re-run constantly while it evolves); and Spark is left doing the windowed
joins it is actually good at. The cost is an intermediate dataset on disk.

## Known limitations, in one place

1. **Duration quantization of one polling interval**, currently ±60s. Aggregate-safe,
   not per-row precise. `arrival_bracket_sec` carries the real figure per row; see
   `polling-cadence.md`.
2. **Dwell time is inside `actual_duration_sec`.** No intermediate departures published.
3. **Repeat runs share one timetable** (17.9% of rows), so their `delay_sec` is
   approximate. Flagged by `trip_run > 0`.
4. **Schedule version may not match** the realtime feed. Flagged per row by
   `schedule_version_agrees`.
5. **~8.9% of stop traversals are unobserved** by VehiclePositions; where TripUpdates
   cannot fill them the segment spans the gap, recorded as `stop_span > 1`.
6. **`occupancy_status` is unusable** — populated on 91.5% of records and every single
   value is `EMPTY`, during rush hour. The congestion model will need WMATA's separate
   Bus & Rail Crowding feed.
7. **Bus is untested.** The pipeline is mode-parameterised and `--mode bus` will run, but
   every measurement here is rail. Bus uses different `stop_id` formats and is ~10x the
   volume; the field census should be re-run before trusting it.
