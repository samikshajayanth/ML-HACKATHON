
import os
import requests
import streamlit as st
import folium

from streamlit_folium import st_folium

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_URL = os.getenv(
    "PATHSNAP_API_URL",
    "http://127.0.0.1:8001"
)

st.set_page_config(
    page_title="PathSnap | GPS Route Reconstruction",
    page_icon="🗺️",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🗺️ PathSnap")
st.subheader("GPS Trajectory Map Matching & Route Reconstruction")

st.markdown(
    "Reconstruct realistic vehicle routes from noisy GPS observations "
    "using machine learning and Hidden Markov Models."
)

st.divider()

# --------------------------------------------------
# BACKEND CONNECTION
# --------------------------------------------------

st.markdown("### Backend Connection")

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )

    health_response.raise_for_status()

    st.success("Backend connected successfully!")

except requests.RequestException as error:
    st.error("Backend connection failed.")
    st.code(str(error))
    st.info(f"API address: {API_URL}")
    st.stop()

# --------------------------------------------------
# FETCH AVAILABLE TRIPS
# --------------------------------------------------

st.markdown("### Select Evaluation Trip")

try:
    trips_response = requests.get(
        f"{API_URL}/trips",
        timeout=10
    )

    trips_response.raise_for_status()

    trips = trips_response.json().get("trips", [])

except requests.RequestException as error:
    st.error("Unable to retrieve trips from the backend.")
    st.code(str(error))
    st.stop()

except ValueError:
    st.error("Backend returned an invalid JSON response.")
    st.stop()

# --------------------------------------------------
# TRIP SELECTION
# --------------------------------------------------

if not trips:

    st.info(
        "No evaluation trips are currently available. "
        "The actual dataset and trip-loading function "
        "still need to be integrated into the backend."
    )

    st.caption(
        "This is an empty-state message, not a prediction result."
    )

    st.stop()

trip_id = st.selectbox(
    "Choose a trip ID",
    trips
)

# --------------------------------------------------
# PREDICTION REQUEST
# --------------------------------------------------

if st.button(
    "🚗 Reconstruct Route",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Processing GPS observations and reconstructing route..."
    ):

        try:

            prediction_response = requests.post(
                f"{API_URL}/predict/{trip_id}",
                timeout=120
            )

            prediction_response.raise_for_status()

            prediction = prediction_response.json()

            st.success("Route reconstruction completed!")

            # ------------------------------------------
            # ROUTE DETAILS
            # ------------------------------------------

            st.markdown("### Prediction Results")

            st.json(prediction)

            # ------------------------------------------
            # FOLIUM MAP
            # ------------------------------------------

            coordinates = prediction.get("coordinates", [])

            if coordinates:

                if (
                    not isinstance(coordinates, list)
                    or len(coordinates) < 2
                ):
                    st.warning(
                        "Not enough route coordinates to draw a map."
                    )

                else:

                    route_map = folium.Map(
                        location=coordinates[0],
                        zoom_start=14
                    )

                    folium.PolyLine(
                        coordinates,
                        color="blue",
                        weight=5,
                        opacity=0.8,
                        tooltip="Reconstructed route"
                    ).add_to(route_map)

                    folium.Marker(
                        coordinates[0],
                        tooltip="Start",
                        icon=folium.Icon(color="green")
                    ).add_to(route_map)

                    folium.Marker(
                        coordinates[-1],
                        tooltip="Destination",
                        icon=folium.Icon(color="red")
                    ).add_to(route_map)

                    st.markdown("### Reconstructed Route Map")

                    st_folium(
                        route_map,
                        width=None,
                        height=550
                    )

            else:

                st.info(
                    "The backend response does not contain route "
                    "coordinates yet. The map will appear after "
                    "the actual decoder is integrated."
                )

            # ------------------------------------------
            # METRICS
            # ------------------------------------------

            metrics = prediction.get("metrics")

            if metrics:

                st.markdown("### Evaluation Metrics")
                st.json(metrics)

            else:

                st.caption(
                    "No evaluation metrics were returned by the backend."
                )

        except requests.HTTPError as error:

            st.error(
                f"Prediction request failed: {error}"
            )

            try:
                st.json(error.response.json())
            except (ValueError, AttributeError):
                pass

        except requests.RequestException as error:

            st.error(
                f"API communication error: {error}"
            )

        except (ValueError, TypeError, KeyError) as error:

            st.error(
                f"Unexpected prediction response: {error}"
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "PathSnap | ML Hackathon | PES University"
)

st.caption(
    "Predictions and metrics are displayed only when returned "
    "by the connected backend."
)
