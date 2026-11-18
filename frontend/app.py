"""PathSnap - Interactive Streamlit Evaluator Interface.

Visualizes GPS observations, candidate road segments, machine learning predictions,
and Hidden Markov Model Viterbi reconstructed routes with Folium mapping.
"""
from __future__ import annotations

import os
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

# -----------------------------------------------------------------------------
# Configuration & Page Setup
# -----------------------------------------------------------------------------
API_URL = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="PathSnap | GPS Route Reconstruction",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .legend-box {
        background-color: #F1F5F9;
        border-radius: 6px;
        padding: 10px 15px;
        font-size: 0.9rem;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown("<div class='main-header'>🗺️ PathSnap</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Distributed Machine Learning & HMM-Based GPS Trajectory Map Matching</div>",
    unsafe_allow_html=True,
)

with st.expander("ℹ️ About PathSnap & Problem Context", expanded=False):
    st.markdown(
        """
        **The Problem:** GPS tracking in urban environments suffers from signal reflection (multipath effects),
        building occlusions, and sensor noise. A vehicle traveling on a street often appears to jump onto parallel streets,
        cross through buildings, or move backwards.

        **PathSnap Solution:**
        1. **Spatial Candidate Generation:** Identifies candidate road arcs within a 100m spatial buffer using projected UTM coordinates and R-Tree indexing.
        2. **Feature Engineering:** Computes perpendicular distance, heading divergence, arc fraction, local speed, and relative distance rankings.
        3. **Machine Learning Classifier:** A trained `HistGradientBoosting` classifier scores candidate segments based on kinematic and geometric features.
        4. **Hidden Markov Model (HMM) & Viterbi Decoding:** Enforces topological road connectivity and transition smoothness using road network shortest paths, resolving disconnected jumps into a coherent continuous trajectory.
        """
    )

# -----------------------------------------------------------------------------
# Sidebar: Backend Connection & Settings
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

default_api_url = os.getenv("PATHSNAP_API_URL", "http://10.30.202.102:8000")
API_URL = st.sidebar.text_input(
    "Backend API URL",
    value=default_api_url,
    help="Enter Laptop 1's IP and port, e.g. http://10.30.202.102:8000",
)

# Check backend health
backend_online = False
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=3)
    if health_resp.status_code == 200 and health_resp.json().get("status") == "healthy":
        backend_online = True
        st.sidebar.success("🟢 Backend API Connected")
    else:
        st.sidebar.warning("🟡 Backend Degraded")
except Exception:
    st.sidebar.error("🔴 Backend Disconnected")
    st.sidebar.info(
        f"Cannot reach `{API_URL}`.\n\n"
        f"1. Make sure Laptop 1 is running: `./start_backend.sh`\n"
        f"2. Ensure both laptops are on the same Wi-Fi.\n"
        f"3. Change the IP above to match Laptop 1."
    )

st.sidebar.divider()
st.sidebar.subheader("🎯 Evaluation Trip Selection")

available_trips = []
if backend_online:
    try:
        trips_resp = requests.get(f"{API_URL}/trips?split=student_test", timeout=5)
        if trips_resp.status_code == 200:
            available_trips = trips_resp.json().get("trips", [])
    except Exception as e:
        st.sidebar.error(f"Error fetching trips: {e}")

if not available_trips:
    st.sidebar.info("Loading default student test trips...")
    available_trips = [
        "00000000", "00000007", "00000009", "00000040", "00000052",
        "00000056", "00000065", "00000076", "00000077", "00000091"
    ]

selected_trip = st.sidebar.selectbox("Select Evaluation Trip (10 Reserved)", available_trips, index=0)

st.sidebar.divider()
st.sidebar.subheader("🔬 Decoder Hyperparameters")

emission_mode = st.sidebar.selectbox(
    "Emission Probability Model",
    options=["hybrid", "ml", "gaussian"],
    index=0,
    help="Hybrid balances learned ML features with physical GPS perpendicular distance.",
)

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    sigma_z = st.number_input("GPS Error σ (m)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
with col_s2:
    beta = st.number_input("Transition β (m)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)

alpha = 0.5
if emission_mode == "hybrid":
    alpha = st.sidebar.slider("ML Weight (α)", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                              help="1.0 = Pure ML, 0.0 = Pure Gaussian Distance")

# -----------------------------------------------------------------------------
# Main Execution Action
# -----------------------------------------------------------------------------
reconstruct_button = st.button("🚀 Reconstruct Route & Run Map Matching", type="primary", use_container_width=True)

if reconstruct_button:
    if not backend_online:
        st.error("Cannot execute: Backend API is not connected. Start the API with `uvicorn backend.main:app --port 8000`.")
        st.stop()

    with st.spinner(f"Matching GPS trajectory for Trip {selected_trip} through ML + HMM Viterbi pipeline..."):
        try:
            url = f"{API_URL}/predict/{selected_trip}"
            params = {
                "split": "student_test",
                "emission_mode": emission_mode,
                "alpha": alpha,
                "sigma_z": sigma_z,
                "beta": beta,
            }
            resp = requests.post(url, params=params, timeout=120)

            if resp.status_code != 200:
                st.error(f"Backend returned error {resp.status_code}: {resp.text}")
                st.stop()

            data = resp.json()
            st.session_state["result"] = data
            st.success(f"Trip {selected_trip} successfully reconstructed in {data['inference_time_s']:.2f} seconds!")

        except requests.exceptions.RequestException as err:
            st.error(f"Failed to communicate with PathSnap API: {err}")
            st.stop()

# -----------------------------------------------------------------------------
# Results Presentation
# -----------------------------------------------------------------------------
if "result" in st.session_state:
    res = st.session_state["result"]
    m = res.get("metrics", {})

    st.markdown("### 📊 Reconstruction Performance Metrics")

    # Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        c_hmm = m.get("continuity_hmm", 0) * 100
        c_ml = m.get("continuity_ml", 0) * 100
        diff = m.get("continuity_improvement_pct", 0)
        st.metric("Route Continuity (HMM)", f"{c_hmm:.1f}%", delta=f"{diff:+.1f}% vs ML Top-1")

    with m_col2:
        c_ml_val = m.get("continuity_ml", 0) * 100
        st.metric("ML Top-1 Continuity", f"{c_ml_val:.1f}%")

    with m_col3:
        acc_hmm = m.get("accuracy_hmm")
        acc_ml = m.get("accuracy_ml")
        if acc_hmm is not None and acc_ml is not None:
            a_diff = m.get("accuracy_improvement_pct", 0)
            st.metric("Route Accuracy (HMM)", f"{acc_hmm * 100:.1f}%", delta=f"{a_diff:+.1f}% vs ML Top-1")
        else:
            st.metric("Route Accuracy", "N/A (Blind Eval)")

    with m_col4:
        cand_recall = m.get("candidate_recall")
        if cand_recall is not None:
            st.metric("Candidate Recall", f"{cand_recall * 100:.1f}%", help="Upper bound on recoverable route arcs")
        else:
            st.metric("Candidate Recall", "N/A")

    with m_col5:
        st.metric("Inference Time", f"{res.get('inference_time_s', 0):.2f}s", f"{res.get('num_observations', 0)} obs")

    st.divider()

    # Map & Details Tabs
    tab_map, tab_comparison, tab_methodology, tab_json = st.tabs([
        "🗺️ Interactive Geographical Map",
        "📈 ML vs HMM Comparison",
        "🧠 Methodology & Theory",
        "🔍 Raw Backend Response",
    ])

    with tab_map:
        st.markdown(
            """
            <div class='legend-box'>
                <b>Map Legend:</b><br/>
                🔵 <b>Blue Solid:</b> Noisy Raw GPS Observations<br/>
                🟢 <b>Green Solid (Thick):</b> PathSnap HMM Viterbi Reconstructed Route (Smooth, Continuous)<br/>
                🟠 <b>Orange Dashed:</b> Independent ML Classifier Top-1 Road Segments<br/>
                🟣 <b>Red/Purple Line:</b> Ground-Truth Reference Route (Server Validation Reference)<br/>
                🟢 <b>Green Marker:</b> Start Point &nbsp;&nbsp;|&nbsp;&nbsp; 🔴 <b>Red Marker:</b> Destination Point
            </div>
            """,
            unsafe_allow_html=True,
        )

        gps_coords = res.get("gps_coordinates", [])
        matched_coords = res.get("matched_coordinates", [])
        ml_coords = res.get("ml_coordinates", [])
        gt_coords = res.get("ground_truth_coordinates")

        if gps_coords:
            center_lat = sum(p[0] for p in gps_coords) / len(gps_coords)
            center_lon = sum(p[1] for p in gps_coords) / len(gps_coords)

            m_folium = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=14,
                tiles="OpenStreetMap",
                control_scale=True,
            )

            # Layer 1: Raw GPS Points
            fg_gps = folium.FeatureGroup(name="Raw GPS Trajectory (Blue)", show=True)
            folium.PolyLine(
                gps_coords,
                color="#2563EB",
                weight=3,
                opacity=0.6,
                tooltip="Raw Noisy GPS Observations",
            ).add_to(fg_gps)
            for i in range(0, len(gps_coords), max(1, len(gps_coords) // 25)):
                folium.CircleMarker(
                    location=gps_coords[i],
                    radius=3,
                    color="#1D4ED8",
                    fill=True,
                    fill_color="#3B82F6",
                    fill_opacity=0.8,
                    tooltip=f"GPS Obs #{i}",
                ).add_to(fg_gps)
            fg_gps.add_to(m_folium)

            # Layer 2: HMM Matched Route
            if matched_coords:
                fg_hmm = folium.FeatureGroup(name="PathSnap HMM Route (Green)", show=True)
                folium.PolyLine(
                    matched_coords,
                    color="#059669",
                    weight=5,
                    opacity=0.9,
                    tooltip="PathSnap Reconstructed Route (Viterbi)",
                ).add_to(fg_hmm)
                fg_hmm.add_to(m_folium)

            # Layer 3: ML Top-1 Route
            if ml_coords:
                fg_ml = folium.FeatureGroup(name="ML Classifier Only (Orange)", show=False)
                folium.PolyLine(
                    ml_coords,
                    color="#F59E0B",
                    weight=3,
                    dash_array="6",
                    opacity=0.8,
                    tooltip="Independent ML Classifier Top-1 Route",
                ).add_to(fg_ml)
                fg_ml.add_to(m_folium)

            # Layer 4: Ground Truth Route
            if gt_coords:
                fg_gt = folium.FeatureGroup(name="Ground-Truth Route (Red)", show=False)
                folium.PolyLine(
                    gt_coords,
                    color="#DC2626",
                    weight=4,
                    dash_array="2, 5",
                    opacity=0.85,
                    tooltip="Reference Ground-Truth Route",
                ).add_to(fg_gt)
                fg_gt.add_to(m_folium)

            # Start and End Markers
            folium.Marker(
                location=gps_coords[0],
                tooltip="Trip Start",
                icon=folium.Icon(color="green", icon="play", prefix="fa"),
            ).add_to(m_folium)

            folium.Marker(
                location=gps_coords[-1],
                tooltip="Trip Destination",
                icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
            ).add_to(m_folium)

            folium.LayerControl(position="topright", collapsed=False).add_to(m_folium)

            st_folium(m_folium, width=None, height=580)
        else:
            st.warning("No GPS coordinates returned to visualize.")

    with tab_comparison:
        st.subheader("Classifier-Only vs. Classifier + HMM Decoding")

        st.markdown(
            """
            | Evaluation Aspect | Independent Classifier (ML Top-1) | Sequential Decoder (PathSnap HMM) | Benefit / Rationale |
            |---|---|---|---|
            | **Route Continuity** | **{ml_cont:.2f}%** | **{hmm_cont:.2f}%** | **+{cont_diff:.2f}%** improvement; resolves topological disconnects |
            | **Route Accuracy** | **{ml_acc}** | **{hmm_acc}** | **{acc_diff}** on ground-truth arcs |
            | **Decision Context** | Observation-local kinematic features only | Joint observation emissions + road shortest path transitions | Prevents unrealistic jumps across parallel streets |
            | **Graph Awareness** | None (predicts each candidate in isolation) | Dijkstra-constrained road graph topological reachability | Guarantees vehicle stays on navigable road segments |
            """.format(
                ml_cont=m.get("continuity_ml", 0) * 100,
                hmm_cont=m.get("continuity_hmm", 0) * 100,
                cont_diff=m.get("continuity_improvement_pct", 0),
                ml_acc=f"{m['accuracy_ml']*100:.2f}%" if m.get("accuracy_ml") is not None else "N/A",
                hmm_acc=f"{m['accuracy_hmm']*100:.2f}%" if m.get("accuracy_hmm") is not None else "N/A",
                acc_diff=f"{m['accuracy_improvement_pct']:+.2f}%" if m.get("accuracy_improvement_pct") is not None else "N/A",
            )
        )

        st.markdown("#### First 20 Matched Road Segments")
        matched_arcs = res.get("matched_arcs", [])
        ml_arcs = res.get("ml_top1_arcs", [])
        n_show = min(20, len(matched_arcs))
        comp_data = []
        for i in range(n_show):
            comp_data.append({
                "Obs Index": i,
                "HMM Matched Arc ID": matched_arcs[i],
                "ML Top-1 Arc ID": ml_arcs[i] if i < len(ml_arcs) else "N/A",
                "Agreement": "✅ Yes" if i < len(ml_arcs) and matched_arcs[i] == ml_arcs[i] else "⚠️ Disagree (HMM Corrected)",
            })
        st.dataframe(comp_data, use_container_width=True)

    with tab_methodology:
        st.subheader("System Architecture & Modeling Choices")
        st.markdown(
            r"""
            ### 1. Emission Probability $P(z_t | c_{t, i})$
            Evaluates how likely GPS point $z_t$ is observed given the vehicle is on candidate road arc $c_{t, i}$:
            - **Gaussian Spatial Model:**
              $$p(z_t | c_{t, i}) = \frac{1}{\sqrt{2\pi}\sigma_z} \exp\left(-\frac{d_i^2}{2\sigma_z^2}\right)$$
              where $d_i$ is the projected perpendicular distance in meters, with default $\sigma_z = 10\text{ m}$.
            - **Machine Learning Likelihood:** Calibrated probabilities $P(c_{t, i} \text{ is correct} | \mathbf{x}_{t, i})$ from `HistGradientBoosting`.
            - **Hybrid Blending:** $\log p_{\text{hybrid}} = \alpha \log p_{\text{ML}} + (1 - \alpha) \log p_{\text{Gauss}}$.

            ### 2. Transition Probability $P(c_{t, j} | c_{t-1, i})$
            Measures the topological plausibility of traveling between candidate $c_{t-1, i}$ and $c_{t, j}$:
            $$p(c_{t, j} | c_{t-1, i}) = \frac{1}{\beta} \exp\left(-\frac{|d_{\text{net}} - \|\Delta z\||}{\beta}\right)$$
            where $d_{\text{net}}$ is the shortest path road-network distance along directed edges and $\|\Delta z\|$ is the observed GPS displacement.
            If $c_{t, j}$ is unreachable from $c_{t-1, i}$ in the road network, a transition penalty is applied.

            ### 3. Viterbi Dynamic Programming Recurrence
            Decodes the global maximum a posteriori (MAP) sequence of road segments:
            $$\log \delta_t(j) = \max_i \left[ \log \delta_{t-1}(i) + \log P(c_{t, j} | c_{t-1, i}) \right] + \log P(z_t | c_{t, j})$$
            Backtracking recovers the most likely contiguous road sequence.
            """
        )

    with tab_json:
        st.subheader("API Response JSON Payload")
        st.json({
            "trip_id": res.get("trip_id"),
            "num_observations": res.get("num_observations"),
            "num_candidates": res.get("num_candidates"),
            "log_likelihood": res.get("log_likelihood"),
            "restarts": res.get("restarts"),
            "metrics": res.get("metrics"),
            "inference_time_s": res.get("inference_time_s"),
        })

else:
    st.info("👈 Select an evaluation trip from the sidebar and click **'Reconstruct Route'** to run the live demonstration.")
