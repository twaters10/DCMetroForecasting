# CI setup: letting GitHub Actions run the ETL

One-time setup for `.github/workflows/etl-daily.yml`. Everything here is AWS console
or GitHub settings work — none of it can be scripted from the repo, which is why it
lives in a document rather than a Makefile.

## Why CI rather than a laptop schedule

A local `launchd` agent was built first and does not work. macOS TCC denies a
LaunchAgent read access to anything under `~/Documents`, where this repo lives, so
executing the pipeline fails before it starts:

```
bash: .../scripts/run_etl_daily.sh: Operation not permitted     # exit 126
```

Probed precisely — a LaunchAgent under `~/Documents` can `chdir` and `stat`, but
cannot list a directory or read a file's contents. Executing a shell script requires
reading it, so it is denied. Pointing the job straight at `.venv/bin/python` does not
help either: Python would then need to read `src/etl/*.py`, same denial.

The fixes were: move the repo out of `~/Documents`, grant Full Disk Access to
`/bin/bash` (broad — every shell script on the machine), or run somewhere without TCC.
CI was chosen. It also removes the "laptop was asleep" failure mode entirely.

`scripts/run_etl_daily.sh` is kept for **manual local runs**, where the terminal does
hold Documents access. CI deliberately does not use it — see below.

## Why OIDC and not access keys

The runner assumes an IAM role via GitHub's OIDC provider, so there are no long-lived
AWS credentials in the repository at all. The alternative — `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` as repo secrets — means a permanent credential that has to be
rotated by hand and is one misconfigured workflow away from being printed to a log.

## Two policies, two different editors — read this before step 2

A role carries two policy documents and the console asks for them in different places.
Mixing them up is the easiest mistake here, and IAM's error messages name the *symptom*
rather than the cause.

| | **Trust** policy | **Permissions** policy |
| --- | --- | --- |
| Answers | *who* may assume the role | *what* the role may do |
| `Principal` | required | **forbidden** |
| `Resource` | absent | **required** |
| Where | generated for you by the Web identity screen; editable afterwards under the role's **Trust relationships** tab | the wizard's **Add permissions** step (step 3 below) |

The single rule: **`Principal` and `Resource` never appear in the same document.** If a
policy has both, or neither, it is in the wrong editor.

## 1. Add GitHub as an IAM identity provider

AWS console → IAM → Identity providers → Add provider → **OpenID Connect**

| Field | Value |
| --- | --- |
| Provider URL | `https://token.actions.githubusercontent.com` |
| Audience | `sts.amazonaws.com` |

One per account. Skip if it already exists.

## 2. Create the role and confirm its trust policy

IAM → Roles → Create role → **Web identity**, selecting the provider above, then fill in:

| Field | Value |
| --- | --- |
| Identity provider | `token.actions.githubusercontent.com` |
| Audience | `sts.amazonaws.com` |
| GitHub organization | `twaters10` |
| GitHub repository | `DCMetroForecasting` |
| GitHub branch | leave as `*` |

"GitHub organization" is the account **owner** — the first path segment of the repo URL.
A personal username goes there exactly as an org name would; there is no separate field
for it.

**Fill in the repository.** Left as `*` the policy becomes `repo:twaters10/*`, letting
any repo you own — including anything you fork later — assume this role.

**There is nothing to paste at this step.** Those three fields *generate* the trust
policy. The JSON below is what you should expect to see afterwards under the role's
**Trust relationships** tab — a verification target, not an input. Pasting it into the
next screen produces the two errors in *Troubleshooting the setup* at the end of this
document.

The `sub` condition is the security boundary: without it, **any** GitHub repository in
the world could assume this role.

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Federated": "arn:aws:iam::767237556899:oidc-provider/token.actions.githubusercontent.com"
    },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
      },
      "StringLike": {
        "token.actions.githubusercontent.com:sub": "repo:twaters10/DCMetroForecasting:*"
      }
    }
  }]
}
```

To restrict further to the default branch only, replace the `sub` value with
`repo:twaters10/DCMetroForecasting:ref:refs/heads/main`. Note this blocks
`workflow_dispatch` runs from other branches, which is usually what you want for a job
that writes to shared storage.

## 3. Attach the permissions policy — the wizard's "Add permissions" step

**This** is the screen you paste JSON into: *Add permissions* → *Create policy* → the
**JSON** tab. Note it has `Resource` and no `Principal`, the opposite of step 2.

The ETL reads `raw/` and `static/`, and writes `processed/`. It has no reason to touch
anything else, and specifically no reason to be able to delete `raw/` — that data is a
live capture and cannot be rebuilt.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListBucketScopedToOurPrefixes",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::metro-pulse-forecast-tawate",
      "Condition": {
        "StringLike": {
          "s3:prefix": ["raw/*", "static/*", "processed/*", ""]
        }
      }
    },
    {
      "Sid": "ReadTheArchive",
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": [
        "arn:aws:s3:::metro-pulse-forecast-tawate/raw/*",
        "arn:aws:s3:::metro-pulse-forecast-tawate/static/*"
      ]
    },
    {
      "Sid": "WriteOnlyProcessed",
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::metro-pulse-forecast-tawate/processed/*"
    }
  ]
}
```

`s3:DeleteObject` on `processed/` is required, not optional:
`processed.sync_partitions` deletes a partition prefix before re-uploading it, because
a re-run can produce a different number of part files and a leftover would be read as
extra rows.

The bare `""` in the `s3:prefix` condition allows the top-level `ListBucket` the
workflow's `aws s3 ls "s3://$S3_BUCKET/"` sanity check performs.

## 4. Set the GitHub repository variables

Settings → Secrets and variables → Actions → **Variables** tab. These are variables,
not secrets — none is sensitive, and having them readable in logs helps debugging.

| Variable | Value |
| --- | --- |
| `AWS_ROLE_ARN` | the role ARN from step 2 |
| `S3_BUCKET` | `metro-pulse-forecast-tawate` |
| `AWS_REGION` | `us-east-1` (optional; the workflow defaults to this) |

No `WMATA_API_KEY` is needed. The ETL never contacts WMATA — it reads static GTFS from
the `static/` archive the collector maintains, precisely so historical data is never
joined against a newer timetable.

## 5. First run

Actions → **etl-daily** → Run workflow. Use `workflow_dispatch` rather than waiting for
the 07:00 UTC schedule.

What a healthy first run looks like:

- `Run tests` — 57 passed
- `Verify identity and configuration` — prints the assumed role ARN, not a user
- `Process outstanding service days` — either processes the outstanding dates or logs
  `nothing outstanding — up to date`. **The latter is success, not a no-op failure**: a
  service day only becomes eligible once it has closed at 04:00 UTC the next day.
- `Report what is now in S3` — one `.parquet` per `service_date=` partition

## Troubleshooting the setup

**`Unsupported Principal: The policy type IDENTITY_POLICY does not support the Principal
element.`** together with **`Missing Resource: Add a Resource or NotResource element to
the policy statement.`**

The step-2 trust policy has been pasted into the step-3 permissions editor. Paste the S3
policy from step 3 there instead; the trust policy needs no pasting at all. Nothing is
wrong with the account or the JSON — see the table near the top of this document.

**`Invalid principal in policy`** when saving a trust policy

The OIDC provider named in `Federated` does not exist in this account, or the account id
is wrong. Check both:

```bash
aws sts get-caller-identity --query Account --output text   # expect 767237556899
aws iam list-open-id-connect-providers
```

**The role saves, but the workflow fails at `Configure AWS credentials`**

That is a run-time failure rather than a setup one — `runbook.txt` §7 covers it. The
usual causes are an unset `AWS_ROLE_ARN` repo variable, a `sub` condition that does not
match this repo, or a missing `permissions: id-token: write` in the workflow.

## Operating notes

**Scheduled workflows are disabled after 60 days of repository inactivity.** GitHub
does this silently. For a project that goes quiet between pushes this is the most
likely way the schedule dies, and nothing will alert you. Either push something
occasionally or check the Actions tab monthly. The catch-up logic means recovery is
just re-enabling and running once — but only while the source data survives its 90-day
retention.

**Scheduled runs can be delayed** by minutes to hours under GitHub load. Harmless here:
the job processes whatever is outstanding rather than assuming it ran on time.

**Only one run at a time**, enforced by the `concurrency` group. A backlog catch-up can
outlast the gap to the next firing, and two runs writing the same `service_date`
partition would race.

**CI does not use `scripts/run_etl_daily.sh`.** That wrapper exports
`AWS_PROFILE=metro-pulse` for local convenience, which on a runner would override the
OIDC credentials the job just assumed and then fail to resolve any profile at all. The
wrapper is for laptops; CI calls `python -m src.etl.catchup` directly.

**Cost.** A rail service day is ~72 seconds of compute locally; the runner spends most
of its time installing dependencies. Well inside the free tier at one run a day, and
the pip cache keeps it that way.
