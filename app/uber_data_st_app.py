# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import re
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import scipy.stats as stats
from geopy.geocoders import Nominatim
import time
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, make_scorer, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
import tempfile, shutil, joblib
import streamlit as st
import os
from pathlib import Path
import altair as alt
import traceback



st.set_page_config(
    page_title="Mobility insights dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)



# Loading predictive models - machine learning (random forest, lasso & linear regression)
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_FILES = {
    "avg_vtat": "avg_vtat_rf_fast.joblib",
    "booking_status": "booking_status_logreg_baseline.joblib",
    "booking_value": "booking_value_ml_model.joblib",}
@st.cache_resource(show_spinner=False)
def _load_model(path: Path):
    return joblib.load(path)
def load_models():
    models = {}
    failed = []
    for key, fname in MODEL_FILES.items():
        path = MODELS_DIR / fname
        try:
            models[key] = _load_model(path)
        except Exception as e:
            failed.append((key, path, str(e)))
    return models, failed

# Models import
models, failed = load_models()
MODELS, MODEL_LOAD_ERRORS = load_models()
# Error message
if failed:
    for key, path, err in failed:
        st.error(f"{key} model import failed — fichier: {path.name} — {err}")
clf_status = models.get("booking_status")
rf_avg_vtat = models.get("avg_vtat")
reg_value   = models.get("booking_value")

#datasets import
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASETS = {
    "raw": "rides_data.csv",
    "clean": "rides_data_clean.csv",
    "delhi": "rides_data_delhi_clean.csv",
    "geocoded": "geocoded_locations.csv",
}
@st.cache_data(show_spinner=False)
def load_csv(path: Path):
    return pd.read_csv(path)
def load_datasets():
    dfs, failed = {}, []
    for key, fname in DATASETS.items():
        path = DATA_DIR / fname
        try:
            dfs[key] = load_csv(path)
        except Exception as e:
            failed.append((key, path, str(e)))
    return dfs, failed
# Chargement
datasets, failed = load_datasets()
# Afficher seulement si échec
if failed:
    for key, path, err in failed:
        st.error(f"❌ Dataset '{key}' failed to load — file: {path.name} — {err}")


# Set title of the app
st.title("Mobility insights dashboard")

# Sidebar title
st.sidebar.title("📈 Dashboard menu")

# Initialisation de la page par défaut
if "page" not in st.session_state:
    st.session_state.page = "home"

# Navigation buttons
if st.sidebar.button("🏠 Home"):
    st.session_state.page = "home"
if st.sidebar.button("🔍 Data exploration"):
    st.session_state.page = "exploration"
if st.sidebar.button("📊 Data analysis"):
    st.session_state.page = "analysis"
if st.sidebar.button("🤖 Predictions"):
    st.session_state.page = "predictions"

    # --- Sidebar footer ---

st.sidebar.markdown("---") 
st.sidebar.markdown(
    """
    **Author**  
    <a href="https://www.linkedin.com/in/lorisdurbano/" target="_blank" style="color:white; text-decoration:none;">
        Loris Durbano
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="16">
    </a>
    """,
    unsafe_allow_html=True
)

#  Contact section
def _stars(n: int) -> str:
    n = int(n); n = max(1, min(5, n))
    return "★" * n + "☆" * (5 - n)

if "show_contact" not in st.session_state:
    st.session_state.show_contact = False

# Contact button
if st.sidebar.button("✉️ Contact"):
    st.session_state.show_contact = not st.session_state.show_contact

# Form
if st.session_state.show_contact:
    with st.sidebar.form("contact_form", clear_on_submit=True):
        message = st.text_area("Message *", height=100, placeholder="Your message…")
        email = st.text_input("Email address *", placeholder="name@example.com")  
        rating = st.radio(
            "Your rating",
            options=[5, 4, 3, 2, 1],
            format_func=_stars,
            horizontal=True,
            index=0,
        )
        sent = st.form_submit_button("Send")

    if sent:
        if not message.strip():
            st.sidebar.error("Please fill the message section.")
        elif not email.strip():
            st.sidebar.error("Please add your email address.")
        elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()):
            st.sidebar.error("Incorrect email address.")
        else:
            fb_dir = Path(__file__).parent / "data"
            fb_dir.mkdir(parents=True, exist_ok=True)
            fb_path = fb_dir / "contact_messages.csv"

            row = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "rating": int(rating),
                "message": message.strip(),
                "email": email.strip(),
            }
            header_needed = not fb_path.exists()
            pd.DataFrame([row]).to_csv(fb_path, mode="a", index=False,
                                    header=header_needed, encoding="utf-8")
            st.sidebar.success("Thank you ! Your message has been submitted ✅")
            st.session_state.show_contact = False

st.sidebar.markdown("---") 

df_rides = datasets.get("raw")  
if df_rides is not None:
    st.sidebar.markdown("**Data source**")
    st.sidebar.download_button(
        label="⬇️ Download the data source",
        data=df_rides.to_csv(index=False).encode("utf-8"),
        file_name="rides_data.csv",
        mime="text/csv"
    )
else:
    st.sidebar.error("❌ 'rides_data.csv' not loaded")

# App pages' content
if st.session_state.page == "home":
    st.header("A data-driven journey through 150,000 Uber trips")
    st.markdown("---")
    st.markdown("""
        What if a few micro-adjustments could boost annual rides and revenue by **3–6%**, while cutting wait times and cancellations?

    This dashboard explores **150,000 Uber rides across India**, combining data exploration, operational diagnostics, and predictive modeling to reveal high-impact levers that improve both efficiency and revenue.

    You’ll discover:
    - 🔥 **When and where to pre-position drivers** to capture the hottest hour × zone peaks  
    - 💰 **How to lift ARPU** through smart micro-upsells and fare tweaks  
    - 🕒 **How to rebalance supply** and monetize shoulder hours in top zones  
    - ⚙️ **How to trigger SLA recovery** when VTAT breaches the 15-minute mark  

    From refining search perimeters to optimizing the vehicle mix, each insight translates data into measurable financial impact —  
    **an estimated +12 000 USD to +24 000 USD per year** in extra revenue, smoother operations, and happier riders. 

    👈 Use the sidebar buttons to navigate through the sections.
    \n Take a look at the dataset's shape :
    """)
    with st.expander("Dataset schema"):
        schema = [
            ("Date", "Date of the booking"),
            ("Time", "Time of the booking"),
            ("Booking ID", "Unique identifier for each ride booking"),
            ("Booking Status", "Status of booking (Completed, Cancelled by Customer, Cancelled by Driver, etc.)"),
            ("Customer ID", "Unique identifier for customers"),
            ("Vehicle Type", "Type of vehicle (Go Mini, Go Sedan, Auto, eBike/Bike, UberXL, Premier Sedan)"),
            ("Pickup Location", "Starting location of the ride"),
            ("Drop Location", "Destination location of the ride"),
            ("Avg VTAT", "Average time for driver to reach pickup location (in minutes)"),
            ("Avg CTAT", "Average trip duration from pickup to destination (in minutes)"),
            ("Cancelled Rides by Customer", "Customer-initiated cancellation flag"),
            ("Reason for cancelling by Customer", "Reason for customer cancellation"),
            ("Cancelled Rides by Driver", "Driver-initiated cancellation flag"),
            ("Driver Cancellation Reason", "Reason for driver cancellation"),
            ("Incomplete Rides", "Incomplete ride flag"),
            ("Incomplete Rides Reason", "Reason for incomplete rides"),
            ("Booking Value", "Total fare amount for the ride"),
            ("Ride Distance", "Distance covered during the ride (in km)"),
            ("Driver Ratings", "Rating given to driver (1-5 scale)"),
            ("Customer Rating", "Rating given by customer (1-5 scale)"),
            ("Payment Method", "Method used for payment (UPI, Cash, Credit Card, Uber Wallet, Debit Card)"),
        ]
        df_schema = pd.DataFrame(schema, columns=["Column Name", "Description"])
        st.dataframe(df_schema, use_container_width=True, hide_index=True)


elif st.session_state.page == "exploration":
    st.header("Data exploration")
    st.markdown("---")
# Cleaning steps summary
    with st.expander("Data preparation (summary)"):
            st.markdown("""
    - **Loaded raw file:** `rides_data.csv`.
    - **Parsed date & time:** converted **Date** to datetime; created **Month**, **Day**, **Weekday**. Parsed **Time** and derived **Hour**.
    - **Standardized columns:** renamed to snake_case for reliable downstream use  
  *(e.g., `Booking ID → booking_id`, `Booking Status → booking_status`, `Vehicle Type → vehicle_type`,  
  `Pickup/Drop Location → pickup_location/drop_location`, `Avg VTAT → avg_vtat`,  
  `Booking Value → booking_value`, `Ride Distance → ride_distance`,  
  `Driver/Customer Rating → driver_rating/customer_rating`, `Payment Method → payment_method`).*
    - **Handled missing values:**  
  - Numerical: `avg_vtat`, `booking_value`, `ride_distance` → **0**  
  - Ratings: `driver_rating`, `customer_rating` → **-1** *(meaning “not rated”)*  
  - Categorical: `payment_method` → **"No payment"**
    - **Duplicates:** none detected.
    - **Outliers:** examined long tail on **booking_value**; **kept all values** (no winsorization) to preserve revenue distribution for analysis.
    """)
    st.markdown(
    """
    After cleaning, the dataset is ready for analysis.  
    Below you can explore the structure of the cleaned dataset, including its columns and value counts.
    """)
    df_clean = datasets.get("clean")
    if df_clean is None:
        st.error("Clean dataset not loaded.")
    else:
        if "Unnamed: 0" in df_clean.columns:
            df_clean = df_clean.drop(columns=["Unnamed: 0"])
    with st.expander("Data schema (cleaned)"):
        schema_df = pd.DataFrame({
            "Column": df_clean.columns,
            "Dtype": df_clean.dtypes.astype(str),
            "Non-Null": df_clean.notna().sum()
        })
        st.dataframe(schema_df, use_container_width=True, hide_index=True)
    with st.expander("Dataset overview: columns & counts"):
        col_info = pd.DataFrame({
            "Column": df_clean.columns,
            "Non-null": df_clean.notna().sum(),
            "Total rows": len(df_clean),
            "Unique values": [df_clean[c].nunique() for c in df_clean.columns]
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)

# Summaries
    st.subheader("Data summary")

    tab_value, tab_status, tab_vehicle, tab_trip, tab_demand, tab_payment, tab_ratings  = st.tabs(["💵 Booking value", "✅ Booking status", "🚘 Vehicle type", 
                                                                                                   "🗺️ Trip characteristics", "⏰ Demand patterns","💳 Payment methods", "⭐ Ratings overview"])
    
    # Booking value summary
    with tab_value:
        st.markdown("Conversion rule: **1 INR = 0.011 USD**. Values < 0 or missing are excluded from the analysis.")

        # find booking value column 
        cols = {c.lower(): c for c in df_clean.columns}
        book_col = (cols.get("booking_value")
        )
        if not book_col:
            st.error("Column for booking value not found (expected 'booking_value').")
            st.stop()

        # Prep data (filter valid, convert to USD, binning) 
        INR_TO_USD = 0.011

        df_val = df_clean[[book_col]].copy()
        df_val = df_val[pd.to_numeric(df_val[book_col], errors="coerce").notna()]
        df_val[book_col] = df_val[book_col].astype(float)
        df_val = df_val[df_val[book_col] >= 0] 

        df_val["booking_value_usd"] = df_val[book_col] * INR_TO_USD

        # bins in USD
        bin_edges = [0, 5, 10, 15, 20, float("inf")]
        bin_labels = ["$0–5", "$6–10", "$11–15", "$16–20", ">$20"]
        df_val["usd_bin"] = pd.cut(
            df_val["booking_value_usd"],
            bins=bin_edges,
            labels=bin_labels,
            right=True, include_lowest=True
        )

        total = len(df_val)
        dist = df_val["usd_bin"].value_counts(dropna=False)
        dist = dist.reindex(bin_labels, fill_value=0)
        dist = dist.rename_axis("Booking value (USD)").reset_index(name="Count")

        dist["Count"] = pd.to_numeric(dist["Count"], errors="coerce").fillna(0).astype(int)
        dist["Percentage (%)"] = (dist["Count"] / max(total, 1) * 100).round(2)

        tab_tbl, tab_cht, tab_ins = st.tabs(["📋 Table", "📊 Chart", "💡 Business Insights"])

        # Table
        with tab_tbl:
            st.subheader("Booking value distribution (USD)")
            st.dataframe(dist[["Booking value (USD)", "Count", "Percentage (%)"]], use_container_width=True, hide_index=True)

        # Chart
        with tab_cht:
            st.subheader("Booking value distribution (USD)")
            chart = (
                alt.Chart(dist)
                .mark_bar()
                .encode(
                    x=alt.X("Booking value (USD):N", sort=bin_labels, title="Booking value (USD)"),
                    y=alt.Y("Count:Q", title="Number of rides"),
                    tooltip=[
                        alt.Tooltip("Booking value (USD):N", title="Booking value"),
                        alt.Tooltip("Count:Q", title="Count", format=",.0f"),
                        alt.Tooltip("Percentage (%):Q", title="Percentage", format=".2f")
                    ],
                )
                .properties(height=360)
                .configure_axis(
                    labelFontSize=13, titleFontSize=14,
                    labelFontWeight="bold", titleFontWeight="bold", grid=True
                )
                .configure_title(fontSize=16, fontWeight="bold")
            )

            st.altair_chart(chart, use_container_width=True)

        # Business insights
        with tab_ins:
            total_rides = int(dist["Count"].sum())
            pct = dict(zip(dist["Booking value (USD)"], dist["Percentage (%)"]))
            cnt = dict(zip(dist["Booking value (USD)"], dist["Count"]))

            tail_pct = (pct.get("$11–15", 0) + pct.get("$16–20", 0) + pct.get(">$20", 0))
            tail_cnt = (cnt.get("$11–15", 0) + cnt.get("$16–20", 0) + cnt.get(">$20", 0))

            st.markdown(f"""

                #### 1️⃣ Demand structure
                • Market is concentrated in rides values between **\$0–5** (**{pct.get("$0–5", 0):.2f}%**, **{cnt.get("$0–5", 0):,} rides**) and **\$6–10** (**{pct.get("$6–10", 0):.2f}%**, **{cnt.get("$6–10", 0):,}**).<br>
                • The higher-value tail (≥ **\$11**) is **small**: **{tail_pct:.2f}%** (**{tail_cnt:,} rides**).

                #### 2️⃣ Business implications
                • **Promotions**: focus on **\$0–10** (≈ **{pct.get("$0–5",0)+pct.get("$6–10",0):.2f}%** of rides) with small discounts (\$0.5–\$1).<br>
                • **Upsell**: prefer **micro-upsells** (tips, low-cost insurance) over expensive add-ons.<br>
                • **Operations**: align **supply** to **short-trip hotspots** to reduce deadhead.

                #### 3️⃣ Checks / further analysis
                • Review rides **> \$20** (airports, anomalies) and compare **revenue share** by bucket.<br>
                • Cross with **payment / vehicle / time** to identify value drivers.

                #### 💡 Key takeaways:
                • **Volume** is in **\$0–10**; **unit value** sits in the small tail **>\$10**.<br>
                • Short-term priority: maximise efficiency and satisfaction on **\$0–5** (~**{pct.get("$0–5", 0):.2f}%** of rides).<br>
                """, unsafe_allow_html=True)

        avg_usd = float(df_val["booking_value_usd"].mean())
        median_usd = float(df_val["booking_value_usd"].median())
        pct_0_5 = float(dist.loc[dist["Booking value (USD)"]=="$0–5", "Percentage (%)"].iloc[0])

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Average booking (USD)", f"{avg_usd:,.2f}")
        c2.metric("Median booking (USD)", f"{median_usd:,.2f}")
        c3.metric("% rides in $0–5", f"{pct_0_5:.0f}%")


    # ✅ Booking Status Summary 
    with tab_status :
        col = "booking_status" if "booking_status" in df_clean.columns else "Booking Status"

        status_counts = (
            df_clean[col]
            .value_counts(dropna=False)
            .rename_axis("booking_status")
            .reset_index(name="count")
        )
        status_counts["percentage"] = (status_counts["count"] / status_counts["count"].sum() * 100).round(1)

        tab_table_bs, tab_chart_bs, tab_insights_bs = st.tabs(["📋 Table", "📊 Chart", "💡 Business Insights"])

        with tab_table_bs:
            st.dataframe(
                status_counts.rename(columns={
                    "booking_status": "Booking Status",
                    "count": "Count",
                    "percentage": "Percentage (%)"
                }),
                use_container_width=True,
                hide_index=True
            )

        with tab_chart_bs:
            chart_bs = alt.Chart(status_counts).mark_bar().encode(
                x=alt.X("count:Q", title="Number of rides"),
                y=alt.Y("booking_status:N", sort="-x", title="Booking status"),
                tooltip=["booking_status", "count", "percentage"]
        ).properties(height=300)

            labels_bs = alt.Chart(status_counts).mark_text(align="left", dx=4).encode(
                x="count:Q",
                y=alt.Y("booking_status:N", sort="-x"),
                text=alt.Text("percentage:Q", format=".1f")
            )

            st.altair_chart(chart_bs + labels_bs, use_container_width=True)

            # Cancellation reasons 
            st.subheader("Cancellation reasons")

            cust_reason_col_candidates = [
            "reason_for_cancelling_by_customer","Reason for cancelling by Customer",]
            driver_reason_col_candidates = ["driver_cancellation_reason","Driver Cancellation Reason",]

            def first_present(cands, cols):
                for c in cands:
                    if c in cols:
                        return c
                return None

            cust_col   = first_present(cust_reason_col_candidates, df_clean.columns)
            driver_col = first_present(driver_reason_col_candidates, df_clean.columns)

            # Subplots
            c1, c2 = st.columns(2)

            TOP_N = 5 

            # Customer cancellations
            with c1:
                if cust_col:
                    cust_counts = (
                        df_clean.loc[df_clean[cust_col].notna(), cust_col]
                        .value_counts()
                        .head(TOP_N)
                        .rename_axis("reason")
                        .reset_index(name="count")
                    )
                    chart_cust = (
                        alt.Chart(cust_counts)
                        .mark_bar()
                        .encode(
                            x=alt.X("count:Q", title="Count"),
                            y=alt.Y("reason:N", sort="-x", title="Customer reason"),
                            tooltip=["reason", "count"],
                        )
                        .properties(height=180)
                    )
                    st.altair_chart(chart_cust, use_container_width=True)
                else:
                    st.info("No customer-cancellation reason column found.")

            # Driver cancellations
            with c2:
                if driver_col:
                    drv_counts = (
                        df_clean.loc[df_clean[driver_col].notna(), driver_col]
                        .value_counts()
                        .head(TOP_N)
                        .rename_axis("reason")
                        .reset_index(name="count")
                    )
                    chart_drv = (
                        alt.Chart(drv_counts)
                        .mark_bar()
                        .encode(
                            x=alt.X("count:Q", title="Count"),
                            y=alt.Y("reason:N", sort="-x", title="Driver reason"),
                            tooltip=["reason", "count"],
                        )
                        .properties(height=180)
                    )
                    st.altair_chart(chart_drv, use_container_width=True)
                else:
                    st.info("No driver-cancellation reason column found.")

        with tab_insights_bs:
            total_rides = int(status_counts["count"].sum())

            def pct(name: str) -> float:
                row = status_counts.loc[
                    status_counts["booking_status"].str.lower() == name.lower(), "percentage"
                ]
                return float(row.iloc[0]) if not row.empty else 0.0

            completion_rate = pct("Completed")
            cancel_customer = pct("Cancelled by Customer")
            cancel_driver   = pct("Cancelled by Driver")
            no_driver_found = pct("No Driver Found")
            incomplete_rate = pct("Incomplete")
            total_cancel_pct = cancel_customer + cancel_driver + no_driver_found
            plus1pp_completed = int(round(total_rides * 0.01))

            frictions = {
                "Customer cancellations": cancel_customer,
                "Driver cancellations": cancel_driver,
                "No driver found": no_driver_found
            }
            top_frict_label, top_frict_val = max(frictions.items(), key=lambda x: x[1])

            st.markdown(
                f"""
        #### 1️⃣ Completion is the majority
        **{completion_rate:.1f}%** of rides are completed.
        #### 2️⃣ Cancellations matter: 
        **{total_cancel_pct:.1f}%** of requests end in cancellation — split **{cancel_customer:.1f}% customer**, **{cancel_driver:.1f}% driver**, **{no_driver_found:.1f}% no-match**.
        **Quick lever** : +1 pp completion ≈ **+{plus1pp_completed}** additional completed rides.
        #### 3️⃣ Primary friction 
        **{top_frict_label}** (≈ **{top_frict_val:.1f}%**). Prioritize ETA, matching, pricing window.
        #### 4️⃣ Data quality note
        “Incomplete” = **{incomplete_rate:.1f}%**; monitor for retries/app/network issues.
        #### 💡 Key takeaways:
        - Driver cancellations are the main friction (~18%); reducing them by even 2–3 pp would materially lift completion. 
        - Focus on ETA accuracy and supply placement at peak times to curb drop-offs; keep “Incomplete” under watch with better retry/error handling.
            """
            )

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total rides", int(status_counts["count"].sum()))
        completed = status_counts.loc[status_counts["booking_status"]=="Completed","percentage"]
        col2.metric("Completion rate", f"{float(completed.iloc[0]) if not completed.empty else 0:.1f}%")
        cancell = status_counts.loc[status_counts["booking_status"].str.contains("Cancel", case=False, na=False),"percentage"].sum()
        col3.metric("Cancellation rate", f"{cancell:.1f}%")
    pass

    # 🚘 Vehicle Type Summary
    with tab_vehicle:

        vt = (
        df_clean.groupby("vehicle_type", dropna=False)
        .agg(
            total_bookings=("booking_id", "count"),
            success_rate=("booking_status", lambda x: (x.eq("Completed").mean() * 100).round(1)),
            avg_distance_km=("ride_distance", "mean"),
            total_distance_km=("ride_distance", "sum"),
        )
        .reset_index()
        )

        vt_table = vt.copy()
        vt_table["avg_distance_km"] = vt_table["avg_distance_km"].round(2)
        vt_table["total_distance_km"] = vt_table["total_distance_km"].round(0).astype(int)
        vt_table = vt_table.rename(columns={
        "vehicle_type": "Vehicle Type",
        "total_bookings": "Total Bookings",
        "success_rate": "Success Rate (%)",
        "avg_distance_km": "Avg Distance (km)",
        "total_distance_km": "Total Distance (km)",
        })

        tab_table, tab_chart, tab_insights = st.tabs(["📋 Table", "📊 Chart", "💡 Business Insights"])

        with tab_table:
            st.subheader("Vehicle mix and performance by product line")
            st.markdown("**Success rate** = share of completed trips.")
            st.dataframe(vt_table, use_container_width=True, hide_index=True)

        with tab_chart:
            chart = alt.Chart(vt).mark_bar().encode(
                x=alt.X("total_bookings:Q", title="Total bookings"),
                y=alt.Y("vehicle_type:N", sort="-x", title="Vehicle type"),
                color=alt.Color("success_rate:Q", title="Success rate (%)"),
                tooltip=[
                    alt.Tooltip("vehicle_type:N", title="Vehicle type"),
                    alt.Tooltip("total_bookings:Q", title="Total bookings"),
                    alt.Tooltip("success_rate:Q", title="Success rate (%)", format=".1f"),
                    alt.Tooltip("avg_distance_km:Q", title="Avg distance (km)", format=".2f"),
                    alt.Tooltip("total_distance_km:Q", title="Total distance (km)", format=".0f"),
                ],
            ).properties(height=320)
            st.altair_chart(chart, use_container_width=True)

        with tab_insights:
            total = vt["total_bookings"].sum()
            top3 = vt.sort_values("total_bookings", ascending=False).head(3)
            top3_share = 100 * top3["total_bookings"].sum() / total

            sr_min, sr_max = vt["success_rate"].min(), vt["success_rate"].max()
            dist_mean = vt["avg_distance_km"].mean()
            dist_min, dist_max = vt["avg_distance_km"].min(), vt["avg_distance_km"].max()

            # Share & +1pp impact for top product
            top_row = vt.loc[vt["total_bookings"].idxmax()]
            top_name = str(top_row["vehicle_type"])
            top_plus1pp = int(round(top_row["total_bookings"] * 0.01))

            # Uber XL niche 
            xl_row = vt[vt["vehicle_type"].str.contains("XL", case=False, na=False)]
            xl_share = 100 * xl_row["total_bookings"].sum() / total if not xl_row.empty else 0

            st.markdown(
                f"""
        #### 1️⃣ Volume is concentrated
        **The top 3 vehicle types** account for **{top3_share:.0f}%** of all bookings — they drive most KPIs and should be the **priority** for ops/pricing actions.
        #### 2️⃣ Success rate is fairly homogeneous
        Range **{sr_min:.1f}%–{sr_max:.1f}%**, pointing to **platform-level frictions** (dispatch, ETA, pricing windows) more than product-specific issues.
        #### 3️⃣ Trip length is similar across products
        Average **{dist_mean:.1f} km** (≈ {dist_min:.1f}–{dist_max:.1f} km), so **revenue impact** is mainly about **volume & price**, not distance.
        **Simple lever:** +1 pp success rate on **{top_name}** ≈ **+{top_plus1pp} completed rides** over the period.
        #### 4️⃣ Long-tail / premium niche
        **Uber XL** represents ~**{xl_share:.1f}%** of volume; to treat as **premium use cases** (airport, weekends) rather than a core growth driver.
        #### 💡 Key takeaways:
        - Focus ops/pricing on the top 3 vehicle types (≈63% of bookings)
        - Lift success by fixing platform-level frictions (dispatch, ETA, pricing windows) rather than product-specific tweaks. 
        - Treat Uber XL as a premium niche (≈3%) for targeted campaigns, while core growth comes from volume + price on the main categories.
            """
            )
     

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Vehicle types", vt["vehicle_type"].nunique())
        best = vt.loc[vt["success_rate"].idxmax()]
        col2.metric("Best success rate", f"{best['success_rate']:.1f}%", help=f"{best['vehicle_type']}")
        top = vt.loc[vt["total_bookings"].idxmax()]
        col3.metric("Top volume", f"{int(top['total_bookings']):,}".replace(",", " "), help=f"{top['vehicle_type']}")

    # 🗺️ Trip characteristics 
    with tab_trip:
        # Column names 
        dist_col = "ride_distance" if "ride_distance" in df_clean.columns else "Ride Distance"
        vtat_col = "avg_vtat"      if "avg_vtat"      in df_clean.columns else "Avg VTAT"

        # Bins
        # Distance (km)
        dist_bins = [0, 5, 10, 15, 20, 30, 50, 100, float("inf")]
        dist_lbls = ["0–5", "6–10", "11–15", "16–20", "21–30", "31–50", "51–100", "100+"]

        # VTAT (minutes)
        vtat_bins = [0, 5, 10, 15, 20, 30, 45, 60, 120, float("inf")]
        vtat_lbls = ["0–5", "6–10", "11–15", "16–20", "21–30", "31–45", "46–60", "61–120", "120+"]

        dist_lbls = ["0–5", "6–10", "11–15", "16–20", "21–30", "31–50", "51–100", "100+"]
        vtat_lbls = ["0–5", "6–10", "11–15", "16–20", "21–30", "31–45", "46–60", "61–120", "120+"]

        df_trip = df_clean[[dist_col, vtat_col]].copy()
        df_trip[dist_col] = pd.to_numeric(df_trip[dist_col], errors="coerce")
        df_trip[vtat_col] = pd.to_numeric(df_trip[vtat_col], errors="coerce")
        df_trip = df_trip.dropna(subset=[dist_col, vtat_col])

        df_trip = df_trip.assign(
            distance_bin = pd.Categorical(
            pd.cut(df_trip[dist_col], bins=dist_bins, labels=dist_lbls,
               right=True, include_lowest=True),
            categories=dist_lbls, ordered=True
        ),
        vtat_bin = pd.Categorical(
            pd.cut(df_trip[vtat_col], bins=vtat_bins, labels=vtat_lbls,
               right=True, include_lowest=True),
        categories=vtat_lbls, ordered=True
        ),
    )
        
        df_trip["distance_bin"] = df_trip["distance_bin"].cat.remove_unused_categories()
        df_trip["vtat_bin"] = df_trip["vtat_bin"].cat.remove_unused_categories()

        tab_trip_table, tab_trip_charts, tab_trip_insights = st.tabs(["📋 Table", "📊 Charts", "💡 Business Insights"])

        # Table
        with tab_trip_table:
            dist_agg = (df_trip.groupby("distance_bin")
                    .agg(count=("distance_bin","size"),
                         avg_vtat_min=(vtat_col,"mean"),
                         avg_distance_km=(dist_col,"mean"))
                    .reset_index())
            dist_agg["percentage"] = (dist_agg["count"] / dist_agg["count"].sum() * 100).round(1)
            dist_agg["avg_vtat_min"] = dist_agg["avg_vtat_min"].round(1)
            dist_agg["avg_distance_km"] = dist_agg["avg_distance_km"].round(2)

            vtat_agg = (df_trip.groupby("vtat_bin")
                    .agg(count=("vtat_bin","size"),
                         avg_vtat_min=(vtat_col,"mean"),
                         avg_distance_km=(dist_col,"mean"))
                    .reset_index())
            vtat_agg["percentage"] = (vtat_agg["count"] / vtat_agg["count"].sum() * 100).round(1)
            vtat_agg["avg_vtat_min"] = vtat_agg["avg_vtat_min"].round(1)
            vtat_agg["avg_distance_km"] = vtat_agg["avg_distance_km"].round(2)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Distance (km)")
                st.dataframe(
                    dist_agg.rename(columns={
                    "distance_bin":"Distance (km)",
                    "count":"Count",
                    "percentage":"Percentage (%)",
                    "avg_vtat_min":"Average time to arrive at trip (VTAT) (min)",
                    "avg_distance_km":"Average distance (km)"
                    }),
                    use_container_width=True, hide_index=True
                )
            with c2:
                st.subheader("Avg VTAT (min)")
                st.dataframe(
                    vtat_agg.rename(columns={
                    "vtat_bin":"VTAT (min)",
                    "count":"Count",
                    "percentage":"Percentage (%)",
                    "avg_vtat_min":"Avg VTAT (min)",
                    "avg_distance_km":"Avg distance (km)"
                    }),
                    use_container_width=True, hide_index=True
                )

        # Charts
        with tab_trip_charts:

            st.subheader("Distributions")
            c1, c2 = st.columns(2)

        # Bar chart distance bins
            with c1:
                chart_dist = alt.Chart(dist_agg).mark_bar().encode(
                    x=alt.X("count:Q", title="Number of rides"),
                    y=alt.Y("distance_bin:N", sort=None, title="Distance (km)"),
                    tooltip=["distance_bin","count","percentage","avg_vtat_min","avg_distance_km"]
                ).properties(height=280)
                st.altair_chart(chart_dist, use_container_width=True)

            # Bar chart vtat bins
            with c2:
                chart_vtat = alt.Chart(vtat_agg).mark_bar().encode(
                    x=alt.X("count:Q", title="Number of rides"),
                    y=alt.Y("vtat_bin:N", sort=None, title="VTAT (min)"),
                    tooltip=["vtat_bin","count","percentage","avg_vtat_min","avg_distance_km"]
                ).properties(height=280)
                st.altair_chart(chart_vtat, use_container_width=True)

            st.subheader("Joint distribution (distance × VTAT)")
            # Heatmap
            joint = (df_trip.groupby(["distance_bin","vtat_bin"])
                    .size().reset_index(name="count"))
            heat = alt.Chart(joint).mark_rect().encode(
                x=alt.X("distance_bin:N", title="Distance (km)", sort=dist_lbls),
                y=alt.Y("vtat_bin:N", title="VTAT (min)", sort=vtat_lbls),
                color=alt.Color("count:Q", title="Count", scale=alt.Scale(scheme="blueorange")),
                tooltip=["distance_bin","vtat_bin","count"]
            ).properties(height=320)
            st.altair_chart(heat, use_container_width=True)

        # Insights
        with tab_trip_insights:
                st.markdown(
                    """

                    #### 1️⃣ Usage patterns  
                     - **2-wheelers (Bike, eBike)** handle **short urban trips**, often below **5 km**, ideal for *first/last-mile mobility* or heavy-traffic zones. They enable **fast rotations**, **low fares**, and **high city coverage**.  
                    - **4-wheelers (Auto, Go Mini, Go Sedan, Premier Sedan, Uber XL)** dominate **medium to long trips**, connecting districts or peri-urban areas, but are more exposed to **traffic congestion** and **driver matching delays**.

                    #### 2️⃣ Performance & pickup dynamics  
                    - **2-wheelers** generally achieve **lower VTAT** (driver arrival time) and **higher reliability** for dense, central areas.  
                    - **4-wheelers** show **slightly higher VTATs** due to traffic and larger dispatch radiuses.  
                        - On the heatmap, most **2-wheel rides** fall into **(0–5 km, 0–10 min)** bins, indicating **fast, frictionless pickups**.  
                        - In contrast, **car segments** have more dispersion across longer distances and 10–20 min VTAT ranges.

                    #### 3️⃣ Profitability logic  
                    - **2-wheelers** = **low margin, high turnover** → profit comes from **volume** and **utilization rate**.  
                    - **4-wheelers** = **higher margin per ride**, but more sensitive to **cancellations and idle time**. Their success rate stability is key to total revenue performance.

                    #### 4️⃣ Strategic opportunities  
                    - **eBike** (≈7% of rides) is an *underused growth lever* — sustainable, efficient, and cheap to scale. Prioritize partnerships with **green mobility incentives** or **corporate commuting programs**.  
                    - **Car fleet optimization** should focus on:
                        - **Predictive dispatch** to anticipate long-distance requests.  
                        - **Dynamic pricing** to manage peri-urban supply-demand balance.  
                        - **Performance-based incentives** (reduce VTAT >15 min).  

                    #### 💡 Key takeaways:
                    - The **2-wheel fleet drives urban responsiveness**, while **4-wheelers drive revenue volume**.  
                    - A dual optimization strategy is needed: *speed & flexibility for 2-wheelers*, *reliability & utilization for 4-wheelers*.
                    """
                )

        # KPIs
        # 1) % rides < 5 km
        short_mask = df_trip["distance_bin"].isin(["0–5"])
        pct_short = 100 * short_mask.mean()
        # 2) % rides with VTAT ≤ 10 min
        fast_mask = df_trip["vtat_bin"].isin(["0–5", "6–10"])
        pct_fast = 100 * fast_mask.mean()
        # 3) VTAT 95th percentile (minutes)
        vtat_p95 = float(df_trip[vtat_col].quantile(0.95))
        # 4) Distance bin with highest average VTAT (where volume is not tiny)
        vol_by_bin = df_trip.groupby("distance_bin").size().rename("count")
        mean_vtat_by_bin = df_trip.groupby("distance_bin")[vtat_col].mean()
        bin_stats = pd.concat([vol_by_bin, mean_vtat_by_bin], axis=1).reset_index()
        bin_stats = bin_stats.sort_values("count", ascending=False)  # gros volumes en haut
        slow_row = bin_stats.iloc[:].sort_values(vtat_col, ascending=False).iloc[0]  # plus lent parmi les + volumineux
        slow_bin = str(slow_row["distance_bin"])
        slow_mean = float(slow_row[vtat_col])

        st.markdown("---")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rides < 5 km", f"{pct_short:.1f}%")
        k2.metric("VTAT ≤ 10 min", f"{pct_fast:.1f}%")
        k3.metric("Customer waiting time (95% of rides)", f"≤ {vtat_p95:.1f} min")
        k4.metric("Slowest distance bin", slow_bin, help=f"Avg VTAT ≈ {slow_mean:.1f} min")

    # ⏰ Demand patterns
    with tab_demand:
        cols = {c.lower(): c for c in df_clean.columns}
        dt_col = None

        cols = {c.lower(): c for c in df_clean.columns}
        hour_col    = cols.get("hour")
        weekday_col = cols.get("weekday")
        month_col   = cols.get("month")
        if not all([hour_col, weekday_col, month_col]):
            st.warning("Need columns 'hour', 'weekday' and 'month' in the clean dataset.")
            st.stop()

        df_time = pd.DataFrame({
            "hour":    pd.to_numeric(df_clean[hour_col], errors="coerce"),
            "weekday": pd.to_numeric(df_clean[weekday_col], errors="coerce"),
            "month":   pd.to_numeric(df_clean[month_col], errors="coerce"),
            }).dropna()

        # Weekdays, month and hours normalization
        df_time["hour"]    = df_time["hour"].astype(int).clip(0, 23)
        df_time["month"]   = df_time["month"].astype(int).clip(1, 12)

        wk_min, wk_max = df_time["weekday"].min(), df_time["weekday"].max()
        if wk_min >= 1 and wk_max <= 7:
            df_time["weekday"] = (df_time["weekday"] - 1).astype(int)
        else:
            df_time["weekday"] = df_time["weekday"].astype(int).clip(0, 6)

        weekday_names = [calendar.day_name[i] for i in range(7)]
        month_names   = [calendar.month_name[i] for i in range(1, 13)]

        # Aggregations
        by_hour = (
            df_time.groupby("hour").size().rename("count").reset_index()
            .assign(percentage=lambda d: (d["count"] / d["count"].sum() * 100).round(1))
        )
        by_weekday = (
            df_time.groupby("weekday").size().rename("count").reset_index()
            .assign(percentage=lambda d: (d["count"] / d["count"].sum() * 100).round(1))
        )
        by_weekday["weekday_name"] = by_weekday["weekday"].map(dict(enumerate(weekday_names)))

        by_month = (
            df_time.groupby("month").size().rename("count").reset_index()
            .assign(percentage=lambda d: (d["count"] / d["count"].sum() * 100).round(1))
        )
        by_month["month_name"] = by_month["month"].map(dict(enumerate(month_names, start=1)))

        tab_dem_table, tab_dem_charts, tab_dem_insights = st.tabs(["📋 Table", "📊 Charts","💡 Business Insights"])

        # Table
        with tab_dem_table:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Rides per hour")
                st.dataframe(
                    by_hour.sort_values("hour").rename(columns={
                        "hour": "Hour", "count": "Count", "percentage": "Percentage (%)"
                    }),
                    use_container_width=True, hide_index=True
                )
            with c2:
                st.subheader("Rides per weekday")
                st.dataframe(
                    by_weekday.sort_values("weekday").rename(columns={
                        "weekday_name": "Weekday", "count": "Count", "percentage": "Percentage (%)"
                    })[["Weekday", "Count", "Percentage (%)"]],
                    use_container_width=True, hide_index=True
                )

            st.subheader("Rides per month")
            st.dataframe(
                by_month.sort_values("month").rename(columns={
                    "month_name": "Month", "count": "Count", "percentage": "Percentage (%)"
                })[["Month", "Count", "Percentage (%)"]],
                use_container_width=True, hide_index=True
            )

        # Charts
        with tab_dem_charts:
            st.subheader("Hourly and weekday patterns")
            ch1, ch2 = st.columns(2)

            with ch1:
                chart_hour = alt.Chart(by_hour).mark_bar().encode(
                    x=alt.X("hour:O", title="Hour of day", sort=list(map(str, range(24)))),
                    y=alt.Y("count:Q", title="Number of rides"),
                    tooltip=["hour", "count", "percentage"]
                ).properties(height=280)
                st.altair_chart(chart_hour, use_container_width=True)

            with ch2:
                chart_wd = alt.Chart(by_weekday).mark_bar().encode(
                    x=alt.X("weekday_name:N", title="Weekday", sort=weekday_names),
                    y=alt.Y("count:Q", title="Number of rides"),
                    tooltip=["weekday_name", "count", "percentage"]
                ).properties(height=280)
                st.altair_chart(chart_wd, use_container_width=True)

            st.subheader("Monthly trend")
            chart_month = alt.Chart(by_month).mark_bar().encode(
                x=alt.X("month_name:N", title="Month", sort=month_names),
                y=alt.Y("count:Q", title="Number of rides"),
                tooltip=["month_name", "count", "percentage"]
            ).properties(height=320)
            st.altair_chart(chart_month, use_container_width=True)

        with tab_dem_insights:
            st.markdown(
                """
                #### 1️⃣ Hourly peaks: clear commute patterns  
                Demand shows **two strong peaks** — morning (7–11 AM) and evening (4–8 PM) — consistent with **commuting behavior**.  
                    - Focus **driver repositioning** and **dynamic pricing** on these time slots.  
                    - Balance supply across zones to reduce cancellation rates and pickup delays.

                #### 2️⃣ Low night activity (12–5 AM)  
                Demand drops sharply overnight, with almost no rides before 6 AM. Avoid generic incentives; use **geo-targeted boosts** near airports, nightlife, or logistics hubs.  

                #### 3️⃣ Weekdays are steady — intra-day variation matters more  
                Ride volume is **flat across the week (Mon–Sun)**.  
                    - Operational strategy should be **time-based, not day-based** (hourly surge, hourly SLA).  
                    - Campaigns by hour (e.g., *“Midday Saver Rides”*) will perform better than weekday vs weekend splits.

                #### 4️⃣ Monthly demand stability  
                Rides are **consistent month-to-month**, showing no major seasonality.  
                    - Enables **predictable capacity planning** and **stable driver earnings**.  
                    - Maintain constant performance targets year-round rather than seasonal SLAs.

                #### 5️⃣ Service quality opportunities  
                Peak-hour demand stresses the system: monitor **VTAT** and **cancellation rates** by hour & zone. Deploy **micro-incentives** to drivers when VTAT 14,5 minutes or “No Driver Found” incidents rise.  

                #### 6️⃣ Product mix optimization  
                Promote **2-wheel rides** in dense urban areas during peak hours (faster pickups, cheaper fares). Emphasize **4-wheel options** for longer or off-peak trips to preserve margin.

                #### 💡 Key takeaways:
                Demand patterns are highly predictable — the challenge is not *when* riders request trips,  but **how effectively the fleet adapts to hourly fluctuations** in supply and traffic.
                """
            )

        # --- KPIs ---
        total_rides = len(df_clean)
        peak_hours = list(range(7, 12)) + list(range(16, 21))  # 7–11 AM and 4–8 PM
        peak_share = (df_clean['hour'].isin(peak_hours).sum() / total_rides) * 100

        early_hours = list(range(0, 6))
        early_share = (df_clean['hour'].isin(early_hours).sum() / total_rides) * 100

        avg_rides_per_hour = df_clean.groupby("hour").size().mean().round(0)
        max_rides_hour = df_clean.groupby("hour").size().idxmax()
        max_rides_val = df_clean.groupby("hour").size().max()

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📈 Peak hour share", f"{peak_share:.1f}%", "7–11 AM + 4–8 PM")
        col2.metric("🌙 Night rides share", f"{early_share:.1f}%", "0–5 AM")
        col3.metric("⏰ Avg rides/hour", f"{int(avg_rides_per_hour):,}")
        col4.metric("🚀 Busiest hour", f"{max_rides_hour}h", f"{max_rides_val:,} rides")

    # 💳 Payment methods
    with tab_payment:

            cols = {c.lower(): c for c in df_clean.columns}
            pay_col = (
                cols.get("payment_method")
                or cols.get("payment method")
                or cols.get("paymentmethod")
            )

            if not pay_col:
                st.error("Payment-method column not found (e.g. 'payment_method' or 'Payment Method').")
            else:
                pay_counts = (
                    df_clean[pay_col]
                    .fillna("Unknown")
                    .value_counts(dropna=False)
                    .rename_axis("Payment Method")
                    .reset_index(name="Count")
                )
                pay_counts["Percentage (%)"] = (pay_counts["Count"] / pay_counts["Count"].sum() * 100).round(1)


            tab_table, tab_chart, tab_payment_insights = st.tabs(["📋 Table", "📊 Chart", "💡 Business Insights"])

            # Table
            with tab_table:
                st.dataframe(
                    pay_counts,
                    use_container_width=True,
                    hide_index=True,
                )


            # Chart
            with tab_chart:
                st.markdown("#### Payment method distribution")
                 # Donut chart
                sort_spec = alt.SortField(field="Count", order="descending")
                donut = (
                    alt.Chart(pay_counts)
                    .mark_arc(innerRadius=70, outerRadius=130)
                    .encode(
                        theta=alt.Theta("Count:Q", title="Number of rides"),
                        color=alt.Color("Payment Method:N", title="Payment Method"),
                        order=alt.Order("Count:Q", sort="descending"),
                        tooltip=["Payment Method", "Count", "Percentage (%)"],
                    )
                    .properties(height=380, width=420)
                )
                labels = (
                    alt.Chart(pay_counts)
                    .mark_text(radius=152, size=12, color="white", fontWeight="bold")
                    .encode(
                        theta=alt.Theta("Count:Q", stack=True),
                         order=alt.Order("Count:Q", sort="descending"),            
                        text=alt.Text("Percentage (%):Q", format=".0f")
                    )
                )
                st.altair_chart(donut + labels, use_container_width=True)

            # Business insights
            with tab_payment_insights:
                st.markdown(
                    """

                    #### 1️⃣ What “No payment” means  
                    - **Not a payment option.** It indicates rides where **no transaction was recorded**, typically because the trip was **cancelled** or **incomplete**.  
                    - In this dataset, *No payment* (~**32%**) mirrors the overall **cancellation + incomplete** share, confirming it’s a **proxy for failed trips**, not a tender type.

                    #### 2️⃣ Channel mix (completed rides only)  
                    - **UPI (~31%)** leads digital tenders — consistent with India’s payment behavior.  
                    - **Cash (~17%)** is secondary, suggesting cashless is already the norm.  
                    - **Uber Wallet (~8%)** shows traction; **Cards (~12%)** remain niche vs. UPI.

                    #### 3️⃣ Operational implications
                    - Treat *No payment* as **lost revenue**: monitor it alongside **cancellation reason**, **VTAT/ETA**, and **“No driver found”** by hour/zone.  
                    - **Stabilize UPI**: prioritize reliability/UX (timeouts, retries) since it carries the bulk of paid trips.  
                    - **Grow in-app wallet** with rewards/cashback or subscription bundles to reduce third-party fees and increase stickiness.  
                    - Keep **cash as fallback** but optimize safety/collection flows.

                    #### 4️⃣ Suggested next checks
                    - Build a **conversion funnel**: Bookings → Driver accepted → Trip started → Trip completed (paid) with drop-offs by hour/zone.  
                    - Compare **completion rate / VTAT** for UPI vs. other methods to detect payment-flow friction.  
                    - Track *No payment* trend after ops/pricing changes to quantify recovered revenue.
                    #### 💡 Key takeaways:
                    - “No payment” isn’t a tender type—it’s a proxy for failed/unfinished trips (cancellations + incomplete) and should be tracked with cancellation reasons and VTAT/ETA by hour/zone. 
                    - Prioritize UPI reliability/UX (it drives most paid trips), grow in-app wallet for stickiness/fees, and keep cash as fallback with tighter safety/collection flows.
                    """
                )

#  ⭐ Rating overview
    with tab_ratings:
        st.markdown("Missing values are already replaced by **-1**, "
                "meaning the ride was **not completed** or **not rated**.")

        cols = {c.lower(): c for c in df_clean.columns}
        driver_col = (
            cols.get("driver_rating")
            or cols.get("driver ratings")
            or cols.get("driver rating")
        )
        customer_col = (
            cols.get("customer_rating")
            or cols.get("customer ratings")
            or cols.get("customer rating")
        )

        if not driver_col or not customer_col:
            st.error("Rating columns not found (expecting 'driver_rating' and 'customer_rating').")
            st.stop()

        df_r = df_clean[[driver_col, customer_col]].copy()
        df_r["driver_unrated"] = df_r[driver_col].eq(-1)
        df_r["customer_unrated"] = df_r[customer_col].eq(-1)

        include_unrated = False  # fixed
        decimals = 2             # fixed

        rated_driver = df_r.loc[~df_r["driver_unrated"], driver_col]
        rated_customer = df_r.loc[~df_r["customer_unrated"], customer_col]
        total = len(df_r)

        driver_avg = round(rated_driver.mean(), decimals) if len(rated_driver) else np.nan
        customer_avg = round(rated_customer.mean(), decimals) if len(rated_customer) else np.nan
        driver_unrated_pct = round(100 * df_r["driver_unrated"].mean(), 1)
        customer_unrated_pct = round(100 * df_r["customer_unrated"].mean(), 1)

        tab_table_rt, tab_chart_rt, tab_insights_rt = st.tabs(["📋 Table", "📊 Chart", "💡 Business Insights"])

        # Table
        with tab_table_rt:

            def build_distribution(col_name: str, who_label: str) -> pd.DataFrame:
                possible = [1, 2, 3, 4, 5]  
                counts = df_r[col_name].value_counts(dropna=False)
                out = pd.DataFrame({
                    "rating": possible,
                    "count": [int(counts.get(v, 0)) for v in possible],
                })
                out["percent"] = (out["count"] / total * 100).round(2)
                out["type"] = who_label
                out["rating_label"] = out["rating"].astype(str)
                return out

            dist_driver = build_distribution(driver_col, "Driver")
            dist_customer = build_distribution(customer_col, "Customer")

            table = pd.concat([dist_driver, dist_customer], ignore_index=True)
            table = table[["type", "rating_label", "count", "percent"]].rename(
                columns={"rating_label": "Rating", "count": "Count", "percent": "Percentage (%)"}
            )

            st.subheader("Ratings distribution")
            st.dataframe(table, use_container_width=True, hide_index=True)


        # Chart
        with tab_chart_rt:
            st.subheader("Distribution by rating (Driver vs Customer)")

            chart_data = pd.concat([dist_driver, dist_customer], ignore_index=True)
            x_order = ["1", "2", "3", "4", "5"]

            chart = (
                alt.Chart(chart_data)
                .mark_bar()
                .encode(
                    x=alt.X("rating_label:N", title="Rating", sort=x_order),
                    y=alt.Y("count:Q", title="Number of rides"),
                    color=alt.Color("type:N", title=""),
                    tooltip=[
                        alt.Tooltip("type:N", title="Who"),
                        alt.Tooltip("rating_label:N", title="Rating"),
                        alt.Tooltip("count:Q", title="Count", format=",.0f"),
                        alt.Tooltip("percent:Q", title="Percentage", format=".1f")
                    ],
                )
                .properties(height=360)
                .configure_axis(
                    labelFontSize=13,
                    titleFontSize=14,
                    labelFontWeight="bold",
                    titleFontWeight="bold",
                    grid=True
                )
                .configure_legend(
                    labelFontSize=13,
                    titleFontSize=13,
                    labelFontWeight="bold",
                    titleFontWeight="bold",
                    orient="right"
                )
                .configure_title(
                    fontSize=16,
                    fontWeight="bold"
                )
            )
            st.altair_chart(chart, use_container_width=True)

        # Business insights
        with tab_insights_rt:
            def counts_by(val_col):
                vc = df_r[val_col].value_counts()
                return {r: int(vc.get(r, 0)) for r in [1, 2, 3, 4, 5]}

            drv = counts_by(driver_col)
            cus = counts_by(customer_col)

            total_not_unrated_drv = int((~df_r["driver_unrated"]).sum())
            total_not_unrated_cus = int((~df_r["customer_unrated"]).sum())

            drv_leq3_share = ( (drv[1] + drv[2] + drv[3]) / max(total_not_unrated_drv, 1) ) * 100
            cus_leq3_share = ( (cus[1] + cus[2] + cus[3]) / max(total_not_unrated_cus, 1) ) * 100

            # Markdown narrative (same tone/format as other *Business Insights* tabs)
            st.markdown(
                f"""

        #### 1️⃣ Data unbalanced
        - Ratings are **skewed to the high end**: most scores are **4–5 stars**; very few 1–2.
        - **Unrated share is high** on both sides (**{driver_unrated_pct:.1f}% driver / {customer_unrated_pct:.1f}% customer**) — averages reflect only those who rated.

        #### 2️⃣ Asymmetry (Customer vs Driver)
        - **Customer avg = {customer_avg:.2f}**, **Driver avg = {driver_avg:.2f}** → customers are **more generous**.
        - 5★ volume is higher on the **customer** side (e.g., {cus[5]:,} vs {drv[5]:,}); drivers lean more to **4★**.

        #### 3️⃣ Uncomplete data
        - “≤3★” shares: **Driver {drv_leq3_share:.1f}%** / **Customer {cus_leq3_share:.1f}%** on rated trips.
        - With ~{driver_unrated_pct:.1f}%–{customer_unrated_pct:.1f}% unrated, there’s a **blind spot**: non-ratings may hide neutral/negative experiences.

        #### Potential additionnal analysis / actions needed : 
        - **Lift rating coverage**: in-app nudges H+1, one-tap stars, small rewards → target **<25% unrated**.
        - **Focus on 3★ zone** (early churn signal): trigger a short “reason” micro-survey and coaching playbook for drivers with elevated ≤3★ share.
        - **Segment checks**: compare ratings by **hour/zone/vehicle type/VTAT** to find root causes (e.g., long VTAT → lower customer ratings).
        - **Report** weekly: % rated, % 5★, median rating, and **rating gap (customer − driver)** to track service perception.

        #### 💡 Key takeaways:
        satisfaction is **strong among raters**, but the **coverage gap** (~{driver_unrated_pct:.1f}%/ {customer_unrated_pct:.1f}%) is the biggest lever to de-risk bias and surface improvement areas.
        """
            )
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Driver rating (avg)", driver_avg)
        c2.metric("Customer rating (avg)", customer_avg)
        c3.metric("% unrated (driver / customer)", f"{driver_unrated_pct}% / {customer_unrated_pct}%")


# Recommendations
    try:
        _total_rides = int(total)
    except Exception:
        _total_rides = int(len(df_val))

    try:
        _avg_booking = float(avg_usd)
    except Exception:
        _avg_booking = float(df_val["booking_value_usd"].mean())

    _baseline_year = _avg_booking * _total_rides
    _uplift_low = 0.08 * _baseline_year
    _uplift_high = 0.13 * _baseline_year
    _month_low = _uplift_low / 12.0
    _month_high = _uplift_high / 12.0

    with st.expander("Summary & recommendations", expanded=False):
        st.markdown(f"""
    ### Increasing revenue by **finishing more trips** and **earning a bit more per trip** — without hurting UX

    #### 5 actions
    1) **Reduce driver cancellations**  
    Improve ETA accuracy, tighten pricing windows at peak, and use small targeted bonuses in high-risk zones/hours.

    2) **Convert more “no-match” requests**  
    Proactively reposition drivers before peaks (hour × zone heatmaps).

    3) **Lift average booking slightly**  
    Tune **minimum fare** in the **\$0–5** core (~70% of rides): **+\$0.20–\$0.30** when VTAT is healthy; use micro-surge only where pickup times stay within SLA.

    4) **Improve collection**  
    Stabilize **UPI** (retries/UX), promote **in-app wallet** (cashback/bundles), and monitor **“No payment”** to fix leaks.

    5) **Allocate fleet by use case**  
    **2-wheelers** for short urban trips (speed); **4-wheelers** for longer or off-peak trips (margin).

    #### KPIs to track :
    - **Completion rate**, **Driver cancel %**, **No-match %**  
    - **VTAT** (pickup delay)  
    - **Average booking (USD/ride)** and share **\$0–5 / \$6–10**  
    - **“No payment” rate** and **wallet/UPI share**

    #### Expected impact (based on current dataset scale)
    - Baseline: **{_total_rides:,} rides/year × \${_avg_booking:,.2f} ≈ \${_baseline_year:,.0f}/year**  
    - Uplift :
        - From **+8% to +13%**: **≈ \${_uplift_low:,.0f} to \${_uplift_high:,.0f} per year** *(≈ **\${_month_low:,.0f} to \${_month_high:,.0f} per month**)*
    """)


elif st.session_state.page == "analysis":
    st.header("Data analysis")

    df_clean = datasets.get("clean")
    df_delhi = datasets.get("delhi")
    geo = datasets.get("geocoded")

    if df_clean is None or df_delhi is None:
        st.error("Please load both 'clean' and 'delhi' datasets before entering the analysis page.")
        st.stop()

    # Intro (why Delhi)
    total_all = int(len(df_clean))
    total_delhi = int(len(df_delhi))
    share_delhi = (total_delhi / max(total_all, 1)) * 100

    st.subheader("Scope of analysis: Delhi")
    st.markdown(
        f"We focus this analysis on **Delhi** because it concentrates most trips: ")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total rides (dataset)", f"{total_all:,}")
    m2.metric("Delhi rides", f"{total_delhi:,}")
    m3.metric("Delhi share", f"{share_delhi:.1f}%")

    st.markdown("---")

    show = st.toggle("Show summary", value=False)
    if show:
        st.markdown("""
            **Assumptions:** ~**70,000 rides/year**, **ARPU ≈ $5.6**
            ---
            ### 1) Win the **hour × zone** peaks
            - **🎯 Recommendation:** pre-position drivers in the **3–4 hottest hour / zone cells** and tighten pricing (slightly higher min-fare / fewer promos) during those cells.
            - **🤔 Hypothesis:** better coverage → lower wait & cancellations → more completed rides.
            - **📊 KPIs:** completed rides ↑, VTAT ↓, acceptance rate ↑, cancels/no-match ↓.
            - **💰 Impact (estimation):** +2–4% rides on targeted cells ≈ +7.8 kUSD to + 15.7 kUSD / year.

            ---

            ### 2) Monetize the **$0–5** cluster
            - **🎯 Recommendation:** A/B test a **light minimum fare** and small **micro-upsells** (tips prompts, 0.2–0.5 USD add-ons) **only** where VTAT is short.
            - **🤔 Hypothesis:** tiny amounts at scale lift ARPU without decreasing conversion.
            - **📊 KPIs:** ARPU ↑, share of $0–5 ↓, complaints stable.
            - **💰 Impact:** +1–3% ARPU on ~56% of rides ≈ +2.2 kUSD to +$6.6 kUSD / year.

            ---

            ### 3) Fill **shoulder hours** in **Top-10 zones**
            - **🎯 Recommendation:** small driver incentives + soft promos **outside peaks** in the Top-10 revenue zones.
            - **🤔 Hypothesis:** enough density to monetize off-peak without degrading service.
            - **📊 KPIs:** share of rides in shoulder hours ↑, VTAT stable, Top-10 revenue share ↑.
            - **💰 Impact:** +5–10% rides in those hours ≈ +5 kUSD to +10 kUSD / year (estimation).

            ---

            ### 4) SLA control with triggers (**VTAT > 15’**)
            - **🎯 Recommendation:** auto-trigger **micro-incentives** and adjust the **driver search perimeter** (radius/ETA window) when a cell breaches 15’.
            - **🤔 Hypothesis:** local supply gaps can be corrected quickly.
            - **📊 KPIs:** % cells out of SLA ↓, time-to-recovery ↓, cancels/no-match ↓.
            - **💰 Impact:** +1–2% rides citywide ≈ +3.9 kUSD to +7.8 kUSD / year.

            ---

            ### 5) Match **search perimeter** to **trip length**
            - **🎯 Recommendation:** **<7 km:** tighter perimeter (or shorter ETA cap). **>12 km:** slightly wider perimeter. Prioritize **Auto / Go Mini** on short trips.
            - **🤔 Hypothesis:** Less dead-time → VTAT ↓ and completion ↑.
            - **📊 KPIs:** VTAT by distance bin ↓, completion ↑, idle/empty-km ↓.
            - **💰 Impact:** VTAT −0.5 to −1.0 min + completion +0.5–1 pp ≈ +2 kUSD to +5 kUSD / year (+ ops efficiency).

            ---

            ### 6) Cut **driver cancellations** / **no-match**
            - **🎯 Recommendation:** in red hour×zone cells, use short-term incentives, better ETA previews, and faster re-assignment.
            - **🤔 Hypothesis:** a portion of cancels is responsive to small nudges + clearer info.
            - **📊 KPIs:** driver cancels ↓, no-match ↓, reassign latency ↓, rides ↑.
            - **💰 Impact:** Cancels −2 pp ≈ +1% rides ≈ +3.9 kUSD / year.

            ---

            ### 7) Steer the **vehicle mix** (limit XL, deepen Auto/Mini)
            - **🎯 Recommendation:** reduce XL exposure in low-yield zones; deepen Auto/Mini at peaks; tweak assignment rules accordingly.
            - **🤔 Hypothesis:** better supply–demand fit → utilization ↑ and revenue/type ↑.
            - **📊 KPIs:** utilization by type ↑, rides/driver-hour ↑, revenue per type ↑.
            - **💰 Impact:** +0.5–1.5% net rides ≈ +2 kUSD to +6 kUSD / year (+ idle time ↓).

            ---

            ### 8) Simple **tips & add-ons** loops
            - **🎯 Recommendation:** post-ride **tip reminders** + low-cost **checkout add-ons** on ~30% of rides.
            - **🤔 Hypothesis:** Micro amounts at scale lift ARPU with minimal friction.
            - **📊 KPIs:** upsell take-rate ↑, ARPU ↑, NPS stable.
            - **💰 Impact:** +0.05 to 0.12 USD / ride on 30% of rides ≈ +1.0 kUSD to +2.5 kUSD / year.

            ---

            ### Overall expected lift : combined levers ≈ **+3–6% annual revenue**
                     
            Baseline ≈ **70k rides × $5.6 ≈ $392k/year** → **~+12 kUSD to +24 kUSD/year**, with **lower VTAT** and smoother operations.
                
            ---
                         
                    """)

    with st.expander("Most popular pickup & drop locations in Delhi", expanded=True):
        TOP = 10

        def top_bar_px(df, col, title, scale="Blues"):
            s = (
                df[col].dropna().astype(str)
                .value_counts()
                .head(TOP)
                .rename_axis(col)
                .reset_index(name="Count"))

            fig = px.bar(
                s.sort_values("Count"),
                x="Count", y=col,
                orientation="h",
                text="Count",
                color="Count",                 
                color_continuous_scale=scale,  
                labels={"Count": "Rides", col: ""},
                title=title,)
            
            fig.update_traces(
                texttemplate="%{text:,}",
                textposition="outside",
                cliponaxis=False)
            
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=48, b=20),
                bargap=0.25,
                xaxis=dict(title="Rides", tickformat=","),
                yaxis=dict(title="", categoryorder="total ascending"),
                font=dict(size=13),
                coloraxis_colorbar=dict(title="Rides"))
            return fig

        pu_col = "pickup_location" if "pickup_location" in df_delhi.columns else None
        do_col = "drop_location"   if "drop_location"   in df_delhi.columns else None

        c1, c2 = st.columns(2, gap="large")

        with c1:
            if pu_col:
                fig_pu = top_bar_px(df_delhi, pu_col, "Top 10 pickup locations", scale="Aggrnyl")
                st.plotly_chart(fig_pu, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Pickup location column not found.")

        with c2:
            if do_col:
                fig_do = top_bar_px(df_delhi, do_col, "Top 10 drop locations", scale="Sunset")
                st.plotly_chart(fig_do, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Drop location column not found.")

        # ---------- Toggle for heatmaps ----------
        show_hm = st.checkbox("Show demand heatmaps (Hour × Weekday & Hour × Top-10 pickup zones)", value=False)

        if show_hm:
            # Normalize weekday/hour robustly
            if {"weekday", "hour"}.issubset(df_delhi.columns):
                df_time = df_delhi[["weekday", "hour"]].copy()

                # numeric first
                df_time["weekday"] = pd.to_numeric(df_time["weekday"], errors="coerce")
                df_time["hour"]    = pd.to_numeric(df_time["hour"], errors="coerce")

                # weekday text -> 0..6
                if df_time["weekday"].isna().all():
                    wk = df_delhi["weekday"].astype(str).str.strip().str.lower()
                    map_wk = {"mon":0,"monday":0,"tue":1,"tuesday":1,"wed":2,"wednesday":2,
                            "thu":3,"thursday":3,"fri":4,"friday":4,"sat":5,"saturday":5,"sun":6,"sunday":6}
                    df_time["weekday"] = wk.map(map_wk)

                # hour text ("07:00","7 pm")
                if df_time["hour"].isna().all():
                    h_raw = df_delhi["hour"].astype(str).str.strip()
                    t1 = pd.to_datetime(h_raw, format="%H:%M", errors="coerce")
                    t2 = pd.to_datetime(h_raw, errors="coerce")
                    hour_parsed = t1.fillna(t2)
                    if hour_parsed.notna().any():
                        df_time["hour"] = hour_parsed.dt.hour

                # normalize 1–7 -> 0–6 ; 1–24 -> 0–23
                wk_17 = df_time["weekday"].between(1, 7, inclusive="both")
                if wk_17.any() and not df_time["weekday"].between(0, 6).any():
                    df_time.loc[wk_17, "weekday"] = df_time.loc[wk_17, "weekday"] - 1
                hr_124 = df_time["hour"].between(1, 24, inclusive="both")
                if hr_124.any() and not df_time["hour"].between(0, 23).any():
                    df_time.loc[hr_124, "hour"] = (df_time.loc[hr_124, "hour"] % 24)

                df_time = df_time.dropna()
                df_time = df_time[(df_time["weekday"].between(0,6)) & (df_time["hour"].between(0,23))]
            else:
                df_time = pd.DataFrame()

            # ---------- Row 2: Heatmaps ----------
            h1, h2 = st.columns(2, gap="large")

            # Hour × Weekday
            with h1:
                if df_time.empty:
                    st.info("No interpretable 'weekday'/'hour' to plot the heatmap.")
                else:
                    weekday_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                    hm = (df_time.groupby(["weekday","hour"]).size()
                            .rename("Rides").reset_index())
                    mat = (hm.pivot(index="weekday", columns="hour", values="Rides")
                            .reindex(index=range(7), columns=range(24), fill_value=0))
                    mat.index = [weekday_names[i] for i in mat.index]
                    fig_hw = px.imshow(
                        mat, color_continuous_scale="Viridis", aspect="auto",
                        labels=dict(color="Rides", x="Hour of day", y="Weekday"),
                        title="Heatmap — Hour × Weekday"
                    )
                    fig_hw.update_layout(height=420, margin=dict(l=10, r=10, t=48, b=10), xaxis=dict(dtick=2))
                    st.plotly_chart(fig_hw, use_container_width=True, config={"displayModeBar": False})

            # Hour × Top-10 pickup zones
            with h2:
                if "pickup_location" not in df_delhi.columns:
                    st.info("Column 'pickup_location' not found.")
                elif df_time.empty:
                    st.info("Cannot compute zone heatmap without interpretable hour.")
                else:
                    top_zones = (df_delhi["pickup_location"].dropna().astype(str)
                                .value_counts().head(10).index.tolist())
                    df_zone = df_delhi.loc[df_delhi["pickup_location"].isin(top_zones),
                                        ["pickup_location", "hour"]].copy()
                    df_zone["hour"] = pd.to_numeric(df_zone["hour"], errors="coerce")
                    df_zone = df_zone.dropna()
                    df_zone = df_zone[df_zone["hour"].between(0,23)]

                    if df_zone.empty:
                        st.info("No rows for top pickup zones / hours.")
                    else:
                        hmz = (df_zone.groupby(["pickup_location","hour"]).size()
                                .rename("Rides").reset_index())
                        matz = (hmz.pivot(index="pickup_location", columns="hour", values="Rides")
                                .reindex(index=top_zones, columns=range(24), fill_value=0))
                        fig_zh = px.imshow(
                            matz, color_continuous_scale="Plasma", aspect="auto",
                            labels=dict(color="Rides", x="Hour of day", y="Pickup zone"),
                            title="Heatmap — Hour × Top-10 pickup zones"
                        )
                        fig_zh.update_layout(height=420, margin=dict(l=10, r=10, t=48, b=10), xaxis=dict(dtick=2))
                        st.plotly_chart(fig_zh, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Service quality — VTAT / pickup waiting time", expanded=True):

        vtat_col = next((c for c in ["avg_vtat", "vtat", "pickup_wait", "pickup_wait_minutes"] if c in df_delhi.columns), None)
        if not vtat_col:
            st.info("No VTAT column found (expected: avg_vtat / vtat / pickup_wait / pickup_wait_minutes).")
            st.stop()

        if "hour" not in df_delhi.columns:
            st.info("Column 'hour' (0–23) is required.")
            st.stop()

        base = df_delhi.copy()
        base[vtat_col] = pd.to_numeric(base[vtat_col], errors="coerce")
        base["hour"] = pd.to_numeric(base["hour"], errors="coerce")
        base = base.dropna(subset=[vtat_col, "hour"])
        base = base[base["hour"].between(0, 23)]

        if base.empty:
            st.info("No rows with interpretable VTAT and hour.")
            st.stop()

        # Hour slider (range) 
        hr_min, hr_max = int(base["hour"].min()), int(base["hour"].max())
        hour_range = st.slider("Hour range", 0, 23, (hr_min, hr_max))
        h1, h2 = hour_range
        d = base[(base["hour"] >= h1) & (base["hour"] <= h2)].copy()

        if d.empty:
            st.info("No rides for the selected hour range.")
            st.stop()

        # KPIs (on filtered hour range)
        v = d[vtat_col]
        median_vtat = float(np.nanmedian(v))
        pct_le10 = float((v <= 10).mean() * 100)

        k1, k2 = st.columns(2)
        k1.metric("Median VTAT", f"{median_vtat:.1f} min")
        k2.metric("% pickups ≤ 10 min", f"{pct_le10:.0f}%")

        st.caption("Tip: use micro-incentives on red hours/zones and balance supply accordingly.")

        # ---- Left: VTAT distribution (adaptive bins, hour-filtered) ----
        # Start with default bins, then drop trailing bins that are empty
        default_bins = [0, 5, 10, 15, 20, np.inf]
        default_labels = ["0–5", "6–10", "11–15", "16–20", ">20"]

        dist = (
            pd.cut(v, bins=default_bins, labels=default_labels, include_lowest=True)
            .value_counts()
            .reindex(default_labels, fill_value=0)
            .rename_axis("Pickup wait (min)")
            .reset_index(name="Rides")
        )

        # Drop any trailing zero-bin(s) to declutter (e.g., drop ">20", then "16–20" if both 0)
        while len(dist) > 0 and dist.iloc[-1]["Rides"] == 0:
            dist = dist.iloc[:-1]

        fig_dist = px.bar(
            dist, x="Pickup wait (min)", y="Rides",
            text="Rides", color="Pickup wait (min)",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title=f"VTAT distribution — {h1:02d}:00 to {h2:02d}:59"
        )
        fig_dist.update_traces(texttemplate="%{text:,}")
        fig_dist.update_layout(height=380, margin=dict(l=10, r=10, t=48, b=10), showlegend=False)

        # Right: Map of avg VTAT by pickup zone (bubbles)
        can_map = {"pickup_location", "pickup_latitude", "pickup_longitude"}.issubset(d.columns)

        if can_map:
            zones = (
                d.dropna(subset=["pickup_location", "pickup_latitude", "pickup_longitude"])
                .assign(pickup_latitude=lambda x: pd.to_numeric(x["pickup_latitude"], errors="coerce"),
                        pickup_longitude=lambda x: pd.to_numeric(x["pickup_longitude"], errors="coerce"))
                .dropna(subset=["pickup_latitude", "pickup_longitude"])
                .groupby("pickup_location", as_index=False)
                .agg(
                    avg_vtat=(vtat_col, "mean"),
                    rides=("pickup_location", "count"),
                    lat=("pickup_latitude", "mean"),
                    lon=("pickup_longitude", "mean"),
                )
            )

            if zones.empty:
                map_fig = None
            else:
                map_fig = px.scatter_mapbox(
                    zones,
                    lat="lat", lon="lon",
                    size="rides", size_max=15,
                    color="avg_vtat",
                    color_continuous_scale="RdBu",
                    hover_name="pickup_location",
                    hover_data={"avg_vtat":":.1f", "rides":":,"},
                    zoom=9, height=480,
                    title="Average VTAT by pickup zone (bubble size = rides)"
                )
                map_fig.update_layout(
                    mapbox_style="open-street-map",
                    margin=dict(l=10, r=10, t=48, b=10)
                )
        else:
            map_fig = None

        # Layout: distribution (left) and map (right)
        c1, c2 = st.columns([1, 1.2], gap="large")
        with c1:
            st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})
        with c2:
            if map_fig is None:
                st.info("Map unavailable: need 'pickup_location', 'pickup_latitude', and 'pickup_longitude' columns.")
            else:
                st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Revenue structure — where money comes from", expanded=True):

            req = {"booking_value", "pickup_location", "hour"}
            if not req.issubset(df_delhi.columns):
                st.info(f"Missing columns: {req - set(df_delhi.columns)}")
                st.stop()

            d0 = df_delhi[["booking_value", "pickup_location", "hour"]].copy()
            d0["booking_value"] = pd.to_numeric(d0["booking_value"], errors="coerce")
            d0["hour"] = pd.to_numeric(d0["hour"], errors="coerce")
            d0 = d0.dropna(subset=["booking_value", "pickup_location", "hour"])
            d0 = d0[(d0["hour"].between(0, 23)) & (d0["booking_value"] > 0)]

            if d0.empty:
                st.info("No valid booking values after cleaning.")
                st.stop()

            # KPIs
            avg_usd = float(d0["booking_value"].mean())
            med_usd = float(d0["booking_value"].median())
            pct_0_5 = float((d0["booking_value"].between(0, 5)).mean() * 100)

            rev_by_zone = d0.groupby("pickup_location", as_index=False)["booking_value"].sum() \
                            .sort_values("booking_value", ascending=False)
            rev_share_top5 = (rev_by_zone["booking_value"].head(5).sum() /
                            rev_by_zone["booking_value"].sum() * 100) if not rev_by_zone.empty else 0.0

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Average booking (USD)", f"{avg_usd:.2f}")
            k2.metric("Median booking (USD)", f"{med_usd:.2f}")
            k3.metric("% rides in $0–5", f"{pct_0_5:.0f}%")
            k4.metric("Revenue share (Top-5 zones)", f"{rev_share_top5:.0f}%")

            # 1a) Booking value distribution (histogram)
            bins = [0, 5, 10, 15, 20, np.inf]
            labels = ["$0–5", "$6–10", "$11–15", "$16–20", ">$20"]
            dist = (pd.cut(d0["booking_value"], bins=bins, labels=labels, include_lowest=True)
                    .value_counts().reindex(labels, fill_value=0)
                    .rename_axis("Booking value (USD)").reset_index(name="Rides"))
            fig_dist = px.bar(
                dist, x="Booking value (USD)", y="Rides",
                text="Rides",
                color="Booking value (USD)",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Booking value distribution (USD)"
            )
            fig_dist.update_traces(texttemplate="%{text:,}")
            fig_dist.update_layout(height=360, margin=dict(l=10, r=10, t=48, b=10), showlegend=False)

            # 1b) ARPU by hour
            arpu_hour = d0.groupby("hour", as_index=False)["booking_value"].mean().rename(columns={"booking_value": "arpu"})
            arpu_hour = arpu_hour.set_index("hour").reindex(range(24), fill_value=np.nan).reset_index()
            fig_arpu_hr = px.bar(
                arpu_hour, x="hour", y="arpu",
                color="arpu", color_continuous_scale="Blues",
                labels={"hour": "Hour of day", "arpu": "ARPU (USD)"},
                title="ARPU by hour")
            fig_arpu_hr.update_layout(
                height=360, margin=dict(l=10, r=10, t=48, b=10),
                coloraxis_colorbar=dict(title="ARPU (USD)", thickness=14, len=0.85, ticks="outside"),
                xaxis=dict(dtick=2))

            r1c1, r1c2 = st.columns([1, 1.15], gap="large")
            with r1c1:
                st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})
            with r1c2:
                st.plotly_chart(fig_arpu_hr, use_container_width=True, config={"displayModeBar": False})

            # Top-10 revenue zones — horizontal bar (descending)
            top10_rev = rev_by_zone.head(10).copy()
            top10_rev.rename(columns={"booking_value": "Revenue (USD)"}, inplace=True)
            fig_top10 = px.bar(
                top10_rev.sort_values("Revenue (USD)"),
                x="Revenue (USD)", y="pickup_location",
                orientation="h", text="Revenue (USD)",
                color="Revenue (USD)", color_continuous_scale="Tealgrn",
                title="Top-10 zones by total revenue (USD)")
            fig_top10.update_traces(texttemplate="%{text:.2s}", textposition="outside", cliponaxis=False)
            fig_top10.update_layout(height=420, margin=dict(l=10, r=10, t=48, b=10), showlegend=False)
            st.plotly_chart(fig_top10, use_container_width=True, config={"displayModeBar": False})

            # Zone selector -> ARPU by hour (bar) + Booking distribution
            zone_options = rev_by_zone["pickup_location"].tolist()
            sel_zone = st.selectbox("Zone focus", options=zone_options, index=None, placeholder="— Select a pickup zone —")

            z = d0[d0["pickup_location"] == sel_zone].copy()
            if z.empty:
                st.info("No data for the selected zone.")
                st.stop()

            # KPIs
            z_avg = z["booking_value"].mean()
            z_med = z["booking_value"].median()
            z_pct05 = (z["booking_value"].between(0,5)).mean() * 100
            gk1, gk2, gk3 = st.columns(3)
            gk1.metric(f"{sel_zone} — Avg (USD)", f"{z_avg:.2f}")
            gk2.metric(f"{sel_zone} — Median (USD)", f"{z_med:.2f}")
            gk3.metric(f"{sel_zone} — % $0–5", f"{z_pct05:.0f}%")

            # Zone ARPU by hour
            z_arpu_hr = z.groupby("hour", as_index=False)["booking_value"].mean().rename(columns={"booking_value": "arpu"})
            z_arpu_hr = z_arpu_hr.set_index("hour").reindex(range(24), fill_value=np.nan).reset_index()
            fig_z_arpu = px.bar(
                z_arpu_hr, x="hour", y="arpu",
                color="arpu", color_continuous_scale="Purples",  # DIFFERENT palette
                labels={"hour": "Hour of day", "arpu": "ARPU (USD)"},
                title=f"ARPU by hour — {sel_zone}")
            fig_z_arpu.update_layout(
                height=320, margin=dict(l=10, r=10, t=48, b=10),
                coloraxis_colorbar=dict(title="ARPU (USD)", thickness=14, len=0.85, ticks="outside"),
                xaxis=dict(dtick=2))

            # Zone distribution
            z_dist = (pd.cut(z["booking_value"], bins=bins, labels=labels, include_lowest=True)
                        .value_counts().reindex(labels, fill_value=0)
                        .rename_axis("Bucket (USD)").reset_index(name="Rides"))
            fig_z_dist = px.bar(
                z_dist, x="Bucket (USD)", y="Rides",
                text="Rides",
                color="Bucket (USD)", color_discrete_sequence=px.colors.qualitative.Set3,  # DIFFERENT colors
                title=f"Booking value distribution — {sel_zone}")
            fig_z_dist.update_traces(texttemplate="%{text:,}")
            fig_z_dist.update_layout(height=320, margin=dict(l=10, r=10, t=48, b=10), showlegend=False)

            rz1, rz2 = st.columns([1.15, 1], gap="large")
            with rz1:
                st.plotly_chart(fig_z_arpu, use_container_width=True, config={"displayModeBar": False})
            with rz2:
                st.plotly_chart(fig_z_dist, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Vehicle types — product mix & trip length", expanded=True):

            vt_col  = "vehicle_type" if "vehicle_type" in df_delhi.columns else None
            fare_col = "booking_value" if "booking_value" in df_delhi.columns else None
            dist_col = "ride_distance" if "ride_distance" in df_delhi.columns else None
            vtat_col = next((c for c in ["avg_vtat","vtat","pickup_wait","pickup_wait_minutes"] if c in df_delhi.columns), None)
            status_col = "booking_status" if "booking_status" in df_delhi.columns else None

            need = [vt_col, fare_col]
            if any(c is None for c in need):
                st.info("Missing required columns: vehicle_type and/or booking_value.")
                st.warning()

            d = df_delhi.copy()
            for c in [fare_col, dist_col, vtat_col]:
                if c and c in d.columns:
                    d[c] = pd.to_numeric(d[c], errors="coerce")

            # KPIs 
            # rides per vehicle
            mix = (d.dropna(subset=[vt_col])[vt_col].value_counts().rename_axis(vt_col).reset_index(name="rides"))
            total_rides = int(mix["rides"].sum()) if not mix.empty else 0
            # share of top-3 vehicle types
            share_top3 = (mix.sort_values("rides", ascending=False)["rides"].head(3).sum() / total_rides * 100) if total_rides else 0.0
            # share of short trips < 5 km (if distance available)
            if dist_col and dist_col in d.columns:
                short_share = float((d[dist_col] < 5).mean() * 100)
            else:
                short_share = 0.0

                k1, k2 = st.columns(2)
                k1.metric("Share of top-3 vehicle types", f"{share_top3:.0f}%")
                k2.metric("Share of short trips (<5 km)", f"{short_share:.0f}%")

                st.markdown("---")

                # Rides vs Avg booking & Total revenue (colored bar + revenue line) 
                g = (
                    d.groupby(vt_col, as_index=False)
                    .agg(
                        rides   =(vt_col, "count"),
                        avg_fare=(fare_col, "mean"),
                        revenue =(fare_col, "sum"),))
                g = g.sort_values("rides", ascending=False).reset_index(drop=True)

                # scale revenue for right axis
                g["revenue_k"] = g["revenue"] / 1_000.0 

                fig_mix = make_subplots(specs=[[{"secondary_y": True}]])

                # Bars = rides 
                fig_mix.add_bar(
                    x=g[vt_col],
                    y=g["rides"],
                    marker_color=px.colors.qualitative.Set2[:len(g)],
                    name="Rides",
                    hovertemplate=(
                        "Vehicle: %{x}<br>"
                        "Rides: %{y:,}<br>"
                        "Avg booking: $%{customdata[0]:.2f}<br>"
                        "Total revenue: $%{customdata[1]:,.0f}<extra></extra>"
                    ),
                    )
                
                fig_mix.data[0].customdata = g[["avg_fare", "revenue"]].values

                # Line = revenue (kUSD)
                fig_mix.add_scatter(
                    x=g[vt_col],
                    y=g["revenue_k"],
                    mode="lines+markers+text",
                    name="Revenue (K$)",
                    line=dict(color="#ff7f0e", width=3),
                    marker=dict(size=9),
                    text=g["revenue_k"].round(1).astype(str) + "k",
                    textposition="top center",
                    textfont=dict(size=12),
                    hovertemplate=(
                        "Vehicle: %{x}<br>"
                        "Revenue: %{y:.1f}k USD<br>"
                        "Avg booking: $%{customdata[0]:.2f}<br>"
                        "Total revenue: $%{customdata[1]:,.0f}<extra></extra>"
                    ),
                    secondary_y=True,
                )
                fig_mix.data[1].customdata = g[["avg_fare", "revenue"]].values

                fig_mix.update_layout(
                    title="Rides by vehicle type + revenue (K$)",
                    height=520,
                    margin=dict(l=10, r=10, t=48, b=10),
                    bargap=0.35,
                    xaxis_title="Vehicle type",
                    yaxis_title="Rides",
                    legend=dict(orientation="h", yanchor="top", xanchor="center", x=0.5, y=+0.90, bgcolor="rgba(0,0,0,0)", borderwidth = 0,),
                    font=dict(size=13),
                )

                # Axis
                fig_mix.update_xaxes(tickangle=-15)
                fig_mix.update_yaxes(
                    rangemode="tozero",
                    showgrid=True, gridcolor="rgba(200,200,200,0.2)",
                    tickformat="," 
                )
                fig_mix.update_yaxes(
                    title_text="Revenue (K$)",
                    secondary_y=True,
                    rangemode="tozero",
                    tickformat=".0f"  
                )

                fig_mix.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                fig_mix.update_xaxes(showgrid=False, zeroline=False)
                fig_mix.update_yaxes(showgrid=False, zeroline=False)                  
                fig_mix.update_yaxes(showgrid=False, zeroline=False, secondary_y=True) 

                # Distance vs VTAT (median) 
                c1, c2 = st.columns([1.25, 1], gap="large")
                with c1:
                    st.plotly_chart(fig_mix, use_container_width=True, config={"displayModeBar": False})

                with c2:
                    if (dist_col is None) or (vtat_col is None):
                        st.info("Need both 'ride_distance' and a VTAT column to draw Distance × VTAT.")
                    else:
                        d_dx = d.dropna(subset=[dist_col, vtat_col]).copy()
                        bins = [0, 3, 7, 12, 20, np.inf]
                        labels = ["0–3", "3–7", "7–12", "12–20", ">20"]
                        d_dx["dist_bin"] = pd.cut(d_dx[dist_col], bins=bins, labels=labels, include_lowest=True)
                        vt = (d_dx.groupby("dist_bin")[vtat_col]
                                    .median().reindex(labels).rename("Median VTAT (min)").reset_index())
                        fig_vt = px.bar(
                            vt, x="dist_bin", y="Median VTAT (min)",
                            color="Median VTAT (min)", color_continuous_scale="Mint",
                            title="Distance vs VTAT"
                        )
                        fig_vt.update_layout(height=420, margin=dict(l=10, r=10, t=48, b=10), 
                                            coloraxis_colorbar=dict(title="min"), xaxis_title="Distance", yaxis_title="VTAT (median)")
                        st.plotly_chart(fig_vt, use_container_width=True, config={"displayModeBar": False})


elif st.session_state.page == "predictions":
    st.header("Predictions")
    df_delhi = datasets.get("delhi")
    geo = datasets.get("geocoded")

    #  Helpers 
    def weekday_to_num(name: str) -> int:
        return {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}.get(str(name),0)

    def safe_latlon(name: str, geo_df: pd.DataFrame):
        if geo_df is None or geo_df.empty or not isinstance(name, str):
            return 0.0, 0.0
        cols = {c.lower(): c for c in geo_df.columns}
        loc_col = cols.get("location") or cols.get("name")
        lat_col = cols.get("latitude") or cols.get("lat")
        lon_col = cols.get("longitude") or cols.get("lon")
        if not loc_col or not lat_col or not lon_col:
            return 0.0, 0.0
        row = geo_df[geo_df[loc_col].astype(str).str.lower() == str(name).lower()]
        if row.empty:
            return 0.0, 0.0
        lat = pd.to_numeric(row.iloc[0][lat_col], errors="coerce")
        lon = pd.to_numeric(row.iloc[0][lon_col], errors="coerce")
        return float(lat) if pd.notna(lat) else 0.0, float(lon) if pd.notna(lon) else 0.0

    def get_column_groups_from_pipeline(model):
        prepro = None
        if hasattr(model, "named_steps"):
            for _, step in model.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    prepro = step; break
        if prepro is None and hasattr(model, "steps"):
            for _, step in model.steps:
                if isinstance(step, ColumnTransformer):
                    prepro = step; break
        if prepro is None:
            return set(), set()

        transformers_list = getattr(prepro, "transformers_", None) or prepro.transformers
        numeric_cols, cat_cols = set(), set()

        for _, trf, cols in transformers_list:
            if cols is None: 
                continue
            obj = trf
            if hasattr(trf, "named_steps"):
                for __, step in trf.named_steps.items():
                    obj = step
            if isinstance(obj, OneHotEncoder):
                cat_cols.update(list(cols))
            if isinstance(obj, (StandardScaler, RobustScaler, MinMaxScaler)):
                numeric_cols.update(list(cols))

        return numeric_cols, cat_cols

    def build_base_inputs(df_delhi, pickup_loc, drop_loc, weekday_name, hour, vehicle_type, geo_df):
        pu_lat, pu_lon = safe_latlon(pickup_loc, geo_df)
        do_lat, do_lon = safe_latlon(drop_loc, geo_df)
        mode = lambda col, default: (df_delhi[col].mode(dropna=True).iloc[0]
                                     if col in df_delhi.columns and df_delhi[col].notna().any()
                                     else default)
        return {
            "month":            mode("month", "January"),  
            "day":              mode("day", 1),             
            "weekday":          weekday_name,               
            "hour":             int(hour),                  
            "booking_status":   mode("booking_status", "Completed"),
            "vehicle_type":     vehicle_type,
            "payment_method":   mode("payment_method", "UPI"),
            "ride_distance":    0.0,
            "driver_rating":    0.0,
            "customer_rating":  0.0,
            "reason_for_cancelling_by_customer": np.nan,
            "driver_cancellation_reason":        np.nan,
            "incomplete_rides_reason":           np.nan,
            # GPS
            "pickup_latitude":  pu_lat,
            "pickup_longitude": pu_lon,
            "drop_latitude":    do_lat,
            "drop_longitude":   do_lon,
        }

    def month_name_to_num(m):
        idx = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,}
        if m is None:
            return np.nan
        return idx.get(str(m).strip().lower(), np.nan)

    def month_name_to_num(m):
        idx = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
            "july":7,"august":8,"september":9,"october":10,"november":11,"december":12}
        if m is None: return np.nan
        return idx.get(str(m).strip().lower(), np.nan)

    def weekday_name_to_num(w):
        idx = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}
        if w is None: return np.nan
        return idx.get(str(w).strip().lower(), np.nan)

    def find_column_transformer(model):
        prepro = None
        if hasattr(model, "named_steps"):
            for _, step in model.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    prepro = step; break
        if prepro is None and hasattr(model, "steps"):
            for _, step in model.steps:
                if isinstance(step, ColumnTransformer):
                    prepro = step; break
        return prepro

    def list_cols_by_transformer(prepro: ColumnTransformer):
        numeric_cols, cat_cols, median_cols = set(), set(), set()
        listing = []
        transformers_list = getattr(prepro, "transformers_", None) or prepro.transformers
        for name, trf, cols in transformers_list:
            if cols is None: cols = []
            used_cols = list(cols)
            root = trf
            steps = []
            if isinstance(trf, Pipeline):
                steps = [(n, s.__class__.__name__, getattr(s, "strategy", None)) for n, s in trf.steps]
                for n, s in trf.steps:
                    if isinstance(s, SimpleImputer) and s.strategy == "median":
                        median_cols.update(used_cols)
                root = trf.steps[-1][1]
            else:
                if isinstance(trf, SimpleImputer) and trf.strategy == "median":
                    median_cols.update(used_cols)

            if isinstance(root, OneHotEncoder):
                cat_cols.update(used_cols)
            if isinstance(root, (StandardScaler, RobustScaler, MinMaxScaler)) or any(
                isinstance(s, (StandardScaler, RobustScaler, MinMaxScaler)) for _, s in (trf.steps if isinstance(trf, Pipeline) else [])
            ):
                numeric_cols.update(used_cols)

            listing.append({
                "name": name, "cols": used_cols,
                "is_pipeline": isinstance(trf, Pipeline),
                "steps": steps
            })
        return numeric_cols, cat_cols, median_cols, listing


    def predict_with_model(model, base_dict, feature_overrides=None, debug_label=""):
        if model is None:
            return None, None, ("", "", "")

        features = list(getattr(model, "feature_names_in_", []))
        prepro = find_column_transformer(model)
        if prepro is None:
            raise RuntimeError("No ColumnTransformer inside model.")

        num_cols, cat_cols, median_cols, raw_listing = list_cols_by_transformer(prepro)

        row = {f: base_dict.get(f, np.nan) for f in features}
        if feature_overrides:
            for k, v in feature_overrides.items():
                if k in row: row[k] = v

        if "month" in median_cols and "month" in row:
            row["month"] = month_name_to_num(row["month"])
        if "weekday" in median_cols and "weekday" in row:
            row["weekday"] = weekday_name_to_num(row["weekday"])

        X = pd.DataFrame([row], columns=features).replace({None: np.nan})

        numeric_targets = set(num_cols) | set(median_cols)
        num_intersect = [c for c in X.columns if c in numeric_targets]
        if num_intersect:
            X[num_intersect] = X[num_intersect].apply(pd.to_numeric, errors="coerce").astype("float64")

        for c in [c for c in X.columns if c in cat_cols]:
            ser = X[c].astype("object")
            ser = ser.where(pd.isna(ser), ser.astype(str))
            X[c] = ser

        for c in X.columns:
            if str(X[c].dtype).startswith("string"):
                tmp = X[c].astype("object")
                tmp = tmp.where(pd.isna(tmp), tmp.astype(str))
                X[c] = tmp

        try:
            y = model.predict(X)[0]
        except Exception as ex:
            st.error("Traceback:\n" + traceback.format_exc())
            raise
        return y, X, (sorted(list(num_cols)), sorted(list(cat_cols)), debug_label)

    # UI 
    pu_opts = sorted(df_delhi["pickup_location"].dropna().unique()) if "pickup_location" in df_delhi else []
    do_opts = sorted(df_delhi["drop_location"].dropna().unique()) if "drop_location" in df_delhi else []
    vt_opts = sorted(df_delhi["vehicle_type"].dropna().unique()) if "vehicle_type" in df_delhi else []

    with st.form("predict_both_form", clear_on_submit=False):
        st.subheader("Predictions")

        c1, c2 = st.columns(2)
        with c1:
            pickup_loc = st.selectbox("Pickup location", pu_opts or ["AIIMS"], index=0)
        with c2:
            drop_loc = st.selectbox("Drop location", do_opts or [pickup_loc], index=0)

        c3, c4 = st.columns(2)
        with c3:
            hour = st.slider("Hour of day", 0, 23, 9)
        with c4:
            weekday_name = st.selectbox(
                "Weekday",
                ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
                index=4
            )

        vehicle_type = st.selectbox("Vehicle type", vt_opts or ["Auto"], index=0)

        submitted = st.form_submit_button("Predict")

    #  Predictions
    if submitted:
        try:
            base = build_base_inputs(df_delhi, pickup_loc, drop_loc, weekday_name, hour, vehicle_type, geo)

            # 1) VTAT
            model_vtat = models.get("avg_vtat")
            vtat_pred, X_vtat, dbg_vtat = predict_with_model(
                model_vtat, 
                base, 
                feature_overrides=None,
                debug_label="avg_vtat"
            )

            def format_min_sec(minutes_val):
                """minutes_val: float (minutes). -> (mins, secs, label 'X min YY sec')."""
                if minutes_val is None or (isinstance(minutes_val, float) and np.isnan(minutes_val)):
                    return None, None, None
                try:
                    total_sec = max(0, int(round(float(minutes_val) * 60)))  # minutes -> secondes
                    mins, secs = divmod(total_sec, 60)
                    label = f"{mins} min {secs:02d} sec" if mins else f"{secs} sec"
                    return mins, secs, label
                except Exception:
                    return None, None, None

            vtat_mins, vtat_secs, vtat_label = format_min_sec(vtat_pred)


            # 2) Booking value (USD)
            model_booking = models.get("booking_value")
            booking_pred_usd, X_book, dbg_book = predict_with_model(
                model_booking,
                base,
                feature_overrides={"avg_vtat": float(vtat_pred) if vtat_pred is not None else 0.0},
                debug_label="booking_value"
            )

            st.markdown("### Estimation")
            _, c, _ = st.columns([1, 2, 1])  # centre l'info
            with c:
                if booking_pred_usd is not None:
                    st.metric("Predicted booking value", f"${booking_pred_usd:,.2f} USD")
                else:
                    st.warning("Booking value model unavailable.")


        except Exception as e:
            st.error(f"Prediction error: {e}")
        
        # Map
        try:
            import pydeck as pdk

            pu_lat, pu_lon = safe_latlon(pickup_loc, geo)
            do_lat, do_lon = safe_latlon(drop_loc, geo)

            if all([pu_lat, pu_lon, do_lat, do_lon]) and not any(pd.isna([pu_lat, pu_lon, do_lat, do_lon])):
                points_df = pd.DataFrame([
                    {"name": f"Pickup • {pickup_loc}", "lat": float(pu_lat), "lon": float(pu_lon)},
                    {"name": f"Drop • {drop_loc}",   "lat": float(do_lat), "lon": float(do_lon)},])

                route_df = pd.DataFrame([{
                    "from_lon": float(pu_lon), "from_lat": float(pu_lat),
                    "to_lon":   float(do_lon), "to_lat":   float(do_lat)}])

                layers = [
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=points_df,
                        get_position="[lon, lat]",
                        get_radius=80,
                        pickable=True
                    ),
                    pdk.Layer(
                        "LineLayer",
                         data=route_df,
                         get_source_position="[from_lon, from_lat]",
                         get_target_position="[to_lon, to_lat]",
                         get_width=4,
                    ),
                ]

                view_state = pdk.ViewState(
                    latitude=float(points_df["lat"].mean()),
                    longitude=float(points_df["lon"].mean()),
                    zoom=11,
                    bearing=0,
                    pitch=30,
                )

                st.markdown("### Route map")
                st.pydeck_chart(
                    pdk.Deck(
                        map_style=None,                 
                        initial_view_state=view_state,
                        layers=layers,
                        tooltip={"text": "{name}"}
                    ),
                    use_container_width=True
                )
            else:
                st.info("Error : missing coordinates")

        except Exception as e:
            st.info(f"Map not available ({e}).")



    st.caption("""Warning: external factors not captured in the dataset (traffic, surge pricing, demand peaks, city events, etc.) 
                 are likely stronger drivers of fare / VTAT variation.""")
