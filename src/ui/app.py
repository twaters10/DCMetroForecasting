"""A thin UI over the deployed endpoint.

    AWS_PROFILE=metro-pulse streamlit run src/ui/app.py

Local rather than hosted, deliberately. A SageMaker endpoint requires SigV4, so a static
page cannot call one — the options were a local app, or API Gateway plus a Lambda proxy.
The proxy would add managed compute the project's brief explicitly scoped out ("the only
managed AWS compute in the loop besides the collector Lambda"), so boto3 signs from here
and that claim stays literally true. The trade is that this demos live from a laptop but
is not shareable by URL.

**Warnings are shown, not hidden.** The endpoint says when a prediction is weakly
supported, when recent conditions are stale, and when an arrive-by estimate covers
less of the distribution than its nominal 80%. A UI that rendered every answer with the
same confidence would undo the work that produced those caveats.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ENDPOINT = "metro-pulse-journey"
REGION = "us-east-1"
STATION_INDEX = Path("data/processed/serving/station_index.json")


@st.cache_data
def station_names() -> list[str]:
    """The 98 station names the resolver knows, so the dropdowns cannot be wrong.

    Free-text would let a user type something the endpoint must reject; a dropdown built
    from the same index the endpoint resolves against cannot produce an unknown station.
    """
    index = json.loads(STATION_INDEX.read_text())
    return sorted(set(index["code_to_name"].values()))


@st.cache_resource
def client():
    import boto3

    return boto3.client("sagemaker-runtime", region_name=REGION)


def predict(origin: str, destination: str, when) -> dict:
    payload = {"origin": origin, "destination": destination}
    if when is not None:
        payload["departure_ts"] = when.isoformat()
    response = client().invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read())


def main() -> None:
    st.set_page_config(page_title="Metro Pulse", page_icon="🚇", layout="centered")
    st.title("🚇 How long will my trip take?")
    st.caption("WMATA rail journey times, from a self-built archive of the live feeds.")

    names = station_names()
    left, right = st.columns(2)
    origin = left.selectbox(
        "From", names, index=names.index("Vienna") if "Vienna" in names else 0
    )
    destination = right.selectbox(
        "To", names, index=names.index("Rosslyn") if "Rosslyn" in names else 1
    )

    use_now = st.checkbox("Leaving now", value=True)
    when = None
    if not use_now:
        import datetime as dt

        date = st.date_input("Date", dt.date.today())
        time = st.time_input("Departure time", dt.time(9, 0))
        when = dt.datetime.combine(date, time, tzinfo=dt.UTC)

    if not st.button("Predict", type="primary", use_container_width=True):
        return

    if origin == destination:
        st.warning("Origin and destination are the same.")
        return

    with st.spinner("Asking the model…"):
        try:
            result = predict(origin, destination, when)
        except Exception as error:  # noqa: BLE001 - surface it, do not crash the app
            st.error(f"Could not reach the endpoint: {error}")
            return

    # A refusal is a real answer, not a failure. The resolver declines rather than
    # guessing a platform, and journeys needing a transfer have no single-train answer.
    if "error" in result:
        st.warning(result["error"])
        return

    scheduled = result["scheduled_sec"] / 60
    predicted = result["predicted_min"]
    columns = st.columns(3)
    columns[0].metric("Typical", f"{predicted:.0f} min")
    if result.get("arrive_by_min") is not None:
        columns[1].metric("Budget for", f"{result['arrive_by_min']:.0f} min")
    columns[2].metric(
        "Timetable", f"{scheduled:.0f} min", delta=f"{predicted - scheduled:+.1f} min"
    )

    coverage = result.get("arrive_by_coverage_pct")
    if coverage is not None:
        st.caption(
            f"“Budget for” is the estimate {coverage:.0f}% of these journeys finish "
            "within — measured, not assumed."
        )

    st.caption(
        f"{result['n_segments']} segments · "
        f"{result['origin']} → {result['destination']} · "
        f"run `{result['model_run']}`"
    )

    for warning in result.get("warnings", []):
        st.info(warning, icon="⚠️")

    if not result.get("trustworthy", True):
        st.warning(
            "This model's split is flagged as not trustworthy — treat these "
            "numbers as provisional.",
            icon="🧪",
        )

    with st.expander("Raw response"):
        st.json(result)


if __name__ == "__main__":
    main()
