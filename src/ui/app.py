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

import io
import json
from pathlib import Path

import streamlit as st

# Streamlit executes this file as a TOP-LEVEL SCRIPT, where there is no parent package
# and a relative import raises "attempted relative import with no known parent package".
# The tests import it as `src.ui.app`, where the relative form is the one that works.
# Same dual-context problem as `src/serving/inference.py`, same shape of answer.
try:  # package context: tests, `python -m`
    from . import route_map
except ImportError:  # script context: `streamlit run src/ui/app.py`
    import route_map  # type: ignore[no-redef]

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


def _dark_theme() -> bool:
    """Best effort. A wrong guess costs a slightly off halo colour, not a broken map."""
    try:
        return str(st.get_option("theme.base")).lower() == "dark"
    except Exception:  # noqa: BLE001 - option access varies across Streamlit versions
        return False


def _map_payload(result: dict | None) -> dict:
    """Only the fields the map actually draws.

    Used as the cache key as well as the input, so a rerun that leaves the route alone
    is a cache hit even though the surrounding prediction carries timestamps and
    floats that change on every call.
    """
    if not result or "error" in result:
        return {}
    return {
        "line": result.get("line"),
        "origin_stop_id": result.get("origin_stop_id"),
        "destination_stop_id": result.get("destination_stop_id"),
        "origin": result.get("origin", ""),
        "destination": result.get("destination", ""),
        "legs": [
            {
                k: leg.get(k)
                for k in ("type", "line", "from_stop_id", "to_stop_id", "at")
            }
            for leg in result.get("legs", [])
        ],
    }


@st.cache_data(show_spinner=False)
def _map_png(payload_json: str, dark: bool) -> bytes | None:
    """Render to PNG bytes rather than returning a figure.

    Streamlit re-executes the whole script on every widget interaction, and a warm
    render costs ~50 ms (the first ~650 ms, mostly matplotlib's font cache). Caching
    the bytes makes an unchanged route free. `st.cache_data` cannot hold a Matplotlib
    figure sensibly, and closing the figure in here is also what stops a long session
    accumulating open figures.
    """
    import matplotlib.pyplot as plt

    figure = route_map.render(json.loads(payload_json), dark=dark)
    if figure is None:
        return None
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, transparent=True, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def render_map_panel(result: dict | None) -> None:
    """The right-hand map. Always drawn, with or without a route.

    Before the first prediction this shows the bare network — `route_map.render({})`
    skips the highlight, the labels and the zoom-to-route, leaving the whole system on
    screen. That is deliberate: the panel should not be an empty rectangle on load, and
    drawing it costs nothing because no endpoint call is involved.
    """
    st.markdown("##### Route")

    payload, caption = _map_payload(result), ""
    alternative = (result or {}).get("alternative")

    if alternative:
        # One map, two routes, same frame — which is the only way to see that the
        # direct ride goes the long way round.
        direct_label = f"Direct · {result['predicted_min']:.0f} min"
        alt_label = (
            f"Via {alternative['transfer_station']} · "
            f"{alternative['predicted_min']:.0f} min"
        )
        choice = st.segmented_control(
            "Route shown",
            [direct_label, alt_label],
            default=direct_label,
            label_visibility="collapsed",
        )
        if choice == alt_label:
            payload = _map_payload(
                {
                    **alternative,
                    "origin": result["origin"],
                    "destination": result["destination"],
                }
            )
            caption = (
                f"{alternative['n_segments']} stops, changing at "
                f"{alternative['transfer_station']}"
            )
        else:
            caption = f"{result['n_segments']} stops, no change of train"

    elif result and "error" not in result:
        caption = (
            f"{result['origin']} → {result['destination']} · "
            f"{result['n_segments']} stops"
        )
    else:
        caption = "The whole network. Pick a trip and the route appears here."

    png = _map_png(json.dumps(payload, sort_keys=True), _dark_theme())
    if png is None:
        st.info(
            "Map unavailable — run `python -m src.serving.geometry` to build "
            "`network_geometry.json`.",
            icon="🗺️",
        )
        return
    st.image(png, width="stretch")
    st.caption(caption + " · faded lines are the rest of the network")


def render_breakdown(result: dict) -> None:
    """Split the total into riding, walking and waiting.

    Riding is the part the model predicts and the part that does not change if the
    rider misses their connection; waiting is the part that does. Separating them is
    what makes a tight change legible instead of hidden inside one number.

    A journey with no train change has no walk and no wait, so those read zero — shown
    rather than hidden, so the three numbers always add up to the total on screen.
    """
    columns = st.columns(3)
    columns[0].metric("On trains", f"{result.get('ride_min', 0):.0f} min")
    columns[1].metric("Walking", _short(result.get("walk_sec", 0)))
    columns[2].metric("Waiting", _short(result.get("wait_sec", 0)))


def _short(seconds: float) -> str:
    """Seconds below a minute, minutes to one decimal above it.

    One decimal, not a whole number: these are small quantities, and rounding a
    3.5-minute wait to "4 min" both loses the precision and disagrees with the leg
    detail below, which shows the same wait as 3.5.
    """
    seconds = int(round(seconds))
    if seconds == 0:
        return "none"
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds / 60:.1f} min"


def render_legs(result: dict) -> None:
    """Show a transfer journey leg by leg, with the time on each leg.

    Per leg, not just a combined riding total: "57 minutes on trains" does not tell a
    rider whether that is one long haul and one short hop or an even split, and the
    two feel completely different to sit through. Each leg is also the unit the model
    actually predicts, so a leg is the smallest thing that can be checked against the
    timetable.

    The wait is broken out for the opposite reason — it is the least certain part of
    the estimate, and `if_missed_sec` is what makes a tight change a judgement the
    rider can make rather than a risk hidden inside a total.
    """
    legs = result.get("legs")
    if not legs:
        return

    transfers = result.get("transfers", 0)
    heading = (
        f"**1 train change** · via {result.get('transfer_station', '?')}"
        if transfers == 1
        else f"**{transfers} train changes**"
    )
    st.markdown(heading)

    ride_number = 0
    for leg in legs:
        if leg["type"] == "ride":
            ride_number += 1
            minutes = leg["predicted_sec"] / 60
            scheduled = leg["scheduled_sec"] / 60
            st.markdown(
                f"🚆 **Leg {ride_number}: {leg['from']} → {leg['to']}** — "
                f"**{minutes:.1f} min**"
            )
            st.caption(
                f"{leg['line'].title()} line · {leg['n_segments']} stops · "
                f"timetable {scheduled:.0f} min ({minutes - scheduled:+.1f})"
            )
        else:
            walk = leg["walk_sec"]
            wait = leg["wait_sec"] / 60
            where = (
                f"{walk}s walk between platforms" if walk else "no walk, same platform"
            )
            st.markdown(f"⇄ **Change at {leg['at']}** — **{wait:.1f} min wait**")
            missed = leg.get("if_missed_sec")
            note = f"{where}"
            if missed is not None:
                note += (
                    f" · miss it and the next train is {missed / 60:.0f} min "
                    "from when you reach the platform"
                )
            st.caption(note)


def render_alternative(result: dict) -> None:
    """Show a faster transfer route alongside a direct answer, without replacing it.

    The direct ride stays the headline. A rider who would rather stay in their seat
    should not have that decision made for them by two minutes of arithmetic — but
    they should be told the option exists, because the app cannot tell which they
    would prefer and the timetable alone does not make it obvious.
    """
    alternative = result.get("alternative")
    if not alternative:
        return

    st.divider()
    st.markdown(
        f"#### Faster if you change trains — {alternative['predicted_min']:.0f} min, "
        f"saving about {alternative['saving_min']:.0f}"
    )
    st.caption(
        f"The direct train takes {result['predicted_min']:.0f} min because of the "
        f"route it follows. Changing at {alternative['transfer_station']} covers "
        f"{alternative['n_segments']} stops instead of {result['n_segments']}."
    )
    render_legs(alternative)
    st.caption("Use the toggle above the map to see this route drawn.")


def _run_prediction(origin: str, destination: str, when) -> dict:
    """Call the endpoint, returning a dict that may carry an `error` key.

    Failures are values rather than exceptions so the caller can store one in session
    state and keep rendering the page around it — including the map, which should not
    vanish because a station name was ambiguous.
    """
    if origin == destination:
        return {"error": "Origin and destination are the same."}
    try:
        return predict(origin, destination, when)
    except Exception as error:  # noqa: BLE001 - surface it, do not crash the app
        return {"error": f"Could not reach the endpoint: {error}"}


def render_result(result: dict) -> None:
    """The left column below the controls: the numbers, the legs, the caveats."""
    # A refusal is a real answer, not a failure. The resolver declines rather than
    # guessing a platform, and an unreachable name has no prediction to give.
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
    elif result.get("arrive_by_basis"):
        st.caption(f"“Budget for” is the {result['arrive_by_basis']}.")

    render_breakdown(result)
    render_legs(result)
    render_alternative(result)

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


def main() -> None:
    st.set_page_config(page_title="Metro Pulse", page_icon="🚇", layout="wide")
    st.title("🚇 How long will my trip take?")
    st.caption("WMATA rail journey times, from a self-built archive of the live feeds.")

    controls, mapping = st.columns([5, 6], gap="large")

    with controls:
        names = station_names()
        from_column, to_column = st.columns(2)
        origin = from_column.selectbox(
            "From", names, index=names.index("Vienna") if "Vienna" in names else 0
        )
        destination = to_column.selectbox(
            "To", names, index=names.index("Rosslyn") if "Rosslyn" in names else 1
        )

        use_now = st.checkbox("Leaving now", value=True)
        when = None
        if not use_now:
            import datetime as dt

            date = st.date_input("Date", dt.date.today())
            time = st.time_input("Departure time", dt.time(9, 0))
            when = dt.datetime.combine(date, time, tzinfo=dt.UTC)

        # The result lives in session state because Streamlit re-runs this whole script
        # on EVERY widget interaction. Held in a local, it would be lost the moment the
        # departure time changed, blanking the map for an edit that did not touch it.
        if st.button("Predict", type="primary", width="stretch"):
            with st.spinner("Asking the model…"):
                st.session_state["result"] = _run_prediction(origin, destination, when)

        result = st.session_state.get("result")
        if result is not None:
            render_result(result)

    with mapping:
        # Rendered unconditionally, before any prediction exists. Every branch above
        # used to `return` early, which would now take the map down with it.
        render_map_panel(st.session_state.get("result"))


if __name__ == "__main__":
    main()
