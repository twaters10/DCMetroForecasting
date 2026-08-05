# Static GTFS: the join, and why bundles are archived

The realtime feeds say what actually happened. Static GTFS says what was
*scheduled*. Every `delay_sec` in the processed dataset is the difference, so this
join is load-bearing for the entire project — and it does not work the obvious
way.

All figures below are measured, not assumed. Source: rail snapshots from
`s3://metro-pulse-forecast-tawate/raw/rail_trip_updates/year=2026/month=08/day=05/hour=12/`
(08:00 EDT, 170 entities) against the bundle published 2026-08-05.

## The join is two hops, keyed on `scheduled_trip_id`

Static `trip_id` is a composite: `{scheduled_trip_id}_{schedule_version}`.
Realtime uses the same shape, but carrying whichever version was live when the
snapshot was taken. Joining `trip_id` to `trip_id` therefore matches nothing.

```
realtime trip_id  "11970073_20660"
      │  strip the version suffix → "11970073"
      │  look up trips.scheduled_trip_id
      ▼
trips row         trip_id = "11970073_20670"   (RED, dir 0, service_id 50_R)
      │  join stop_times.txt ON THIS versioned trip_id
      ▼
scheduled times   seq 1 PF_A08_C 17:43:00 · seq 2 PF_A07_C 17:45:00 · …
```

The second hop is the one that is easy to get wrong: **`stop_times.txt` is keyed on
the versioned `trip_id`, not on `scheduled_trip_id`.** Keying it on the stripped
base id returns zero rows. `trips.txt` is the translation table and cannot be
skipped.

| Join attempt | Match rate |
| --- | --- |
| realtime `trip_id` → static `trip_id` | **0 / 170 (0%)** |
| base → `trips.scheduled_trip_id` | **169 / 169 (100%)** |
| … onward to `stop_times` | 169 / 169 |
| realtime `stop_id` → static `stop_id` | 125 / 125 (100%) |

`scheduled_trip_id` is unique across all 14,030 trips, so the join is 1:1 and does
not fan out. `route_id` and `direction_id` agree on every matched trip, with zero
disagreements — the match is real, not coincidental string overlap.

### Exclude non-revenue trips before computing a match rate

The 170th entity is `NR042`, with `route_id == "NR"` — a non-revenue equipment
move. These have no schedule because they are not scheduled service. Counting them
in the denominator understates the match rate (169/170 = 99.4% rather than 100%)
and, worse, makes a genuine future regression indistinguishable from normal
non-revenue traffic. **Filter `route_id == "NR"` first.**

Rail `stop_id` values are platform codes (`PF_A08_C`), not the numeric ids bus
uses. The static join must be validated per mode.

## Why the collector archives bundles daily

WMATA serves only the *current* bundle, and the two modes age very differently:

| | rail | bus |
| --- | --- | --- |
| size | 3.1 MB | 49.8 MB |
| `feed_start_date` → `feed_end_date` | 20260805 → 20260814 (**~10 days**) | 20260621 → 20260912 (~3 months) |
| `feed_version` | *absent* | `S1000250` |
| `calendar.txt` | *absent* (only `calendar_dates.txt`) | present |

A schedule version bump happens *because the timetable changed*. So joining
archived realtime against a later bundle can return the **new** scheduled time
against the **old** actual — a wrong `delay_sec` that looks entirely plausible.
Trips deleted outright just drop out, which is the safe failure and shows up as a
falling match rate. It is the **retimed** trips that corrupt quietly.

Rail's window is about ten days. Once a bundle rotates out it cannot be re-fetched
from any endpoint, so a day not archived is a day whose labels can never be
computed correctly. That is the whole justification for the daily task: not
convenience, but that the alternative is unrecoverable.

The `raw/` prefix additionally carries a 90-day expiry lifecycle rule, which caps
how long any of this stays fixable. `config.py` refuses to start if
`S3_STATIC_PREFIX` nests inside `S3_PREFIX`, because bundles inheriting that
expiry would be deleted on a clock they can never be restored from.

## Realtime and static versions are not in lockstep

This is the subtlety that shapes how the ETL must resolve bundles.

The bundle fetched on 2026-08-05 declares `feed_start_date 20260805`. But realtime
snapshots archived at **08:00 EDT on Aug 5** still emit `_20660` trip_ids — the
previous version. The feed lagged its own published timetable by at least a day.

Two consequences:

1. **"Which bundle was in effect" cannot be resolved from `service_date` alone.**
   The feed window is necessary but not sufficient; the realtime data may still be
   running the previous version's ids.
2. **The `scheduled_trip_id` join is the right key precisely because it is
   version-agnostic.** It matched 100% straight across the version boundary, where
   a `trip_id` join matched 0%.

So the ETL must record **both** identifiers on every output row:

- `static_gtfs_version` — the `feed_start_date` of the bundle actually joined
- `realtime_schedule_version` — the suffix parsed from the realtime `trip_id`

and flag rows where they disagree as having possibly-stale scheduled times. Both
are free to capture at ETL time and impossible to reconstruct afterwards. A
disagreement does not invalidate a row — the 100% match rate above came from
exactly such a pairing — but it is the only signal that a `delay_sec` might be
measured against a timetable that was not in force.

## Archive layout

```
static/{mode}/feed_start=YYYYMMDD/feed_end=YYYYMMDD/{mode}-gtfs-static-{sha256[:12]}.zip
```

Content-addressed, which buys two things:

- **Idempotency without reading S3.** The same bundle hashes to the same key, so
  the daily write is a byte-identical overwrite rather than a duplicate. This is
  not stylistic: the Lambda execution role grants `s3:PutObject` and nothing else,
  so an "already archived?" check would fail with `AccessDenied`. Storage grows per
  *distinct* bundle — roughly one a week for rail, one a quarter for bus.
- **Resolution by listing alone.** The feed window is in the key, so the ETL picks
  the bundle covering a service date from one `list_objects_v2` with no downloads.
  The ETL runs locally under a full-access profile, so listing is available there.

Bundles are stored as the exact bytes WMATA served. They are already compressed, so
unlike the realtime snapshots they are not gzipped again. Objects also carry
`feed-start-date`, `feed-end-date`, `feed-version`, `fetched-at` and `sha256` as S3
metadata; rail's `feed-version` is empty, which is why the key is built from the
feed window rather than from `feed_version`.

## Running it

```bash
# Locally, against the live API, writing to LOCAL_OUTPUT_DIR — no AWS needed.
python infra/lambda_collector/local_run.py --task static-gtfs

# In AWS. The daily EventBridge schedule sends this same payload.
export AWS_PROFILE=metro-pulse
aws lambda invoke --function-name metro-pulse-collector \
  --payload '{"task":"static_gtfs"}' --cli-binary-format raw-in-base64-out /dev/stdout
```

The realtime path is the default: an event with no `task` key, or an unrecognised
one, runs the 60-second collection. That default is deliberate — a malformed event
should never cost a minute of irreplaceable history.
