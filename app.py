import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD CSS
# ============================================================

css_file = Path("style.css")

if css_file.exists():
    st.markdown(
        f"<style>{css_file.read_text()}</style>",
        unsafe_allow_html=True
    )


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path("notebooks/fraud_detection_model.pkl")
DATA_PATH = Path("notebooks/fraudTest.csv")


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model_loaded = False
    model = None


# ============================================================
# LOAD ANALYTICS DATA
# ============================================================

SUMMARY_PATH = Path("analytics_summary.json")

df = None
analytics = None

# ------------------------------------------------------------
# Try full dataset first (local development)
# ------------------------------------------------------------

if DATA_PATH.exists():

    try:
        df = pd.read_csv(DATA_PATH)

    except Exception:
        df = None


# ------------------------------------------------------------
# If full dataset is unavailable, use cloud summary
# ------------------------------------------------------------

if df is None and SUMMARY_PATH.exists():

    try:
        import json

        with open(
            SUMMARY_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            analytics = json.load(file)

    except Exception:
        analytics = None


# ============================================================
# FEATURE ENGINEERING
# ============================================================

if df is not None:

    if "trans_date_trans_time" in df.columns:

        df["trans_date_trans_time"] = pd.to_datetime(
            df["trans_date_trans_time"],
            errors="coerce"
        )

        df["hour"] = (
            df["trans_date_trans_time"]
            .dt.hour
            .fillna(0)
            .astype(int)
        )

        df["month"] = (
            df["trans_date_trans_time"]
            .dt.month
            .fillna(0)
            .astype(int)
        )

        df["day_of_week"] = (
            df["trans_date_trans_time"]
            .dt.dayofweek
            .fillna(0)
            .astype(int)
        )

        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype(int)

        df["is_night"] = (
            (df["hour"] < 6) |
            (df["hour"] >= 22)
        ).astype(int)

    if "amt" in df.columns:

        df["amount_range"] = pd.cut(
            df["amt"],
            bins=[
                0,
                25,
                50,
                100,
                250,
                500,
                1000,
                np.inf
            ],
            labels=[
                "0-25",
                "25-50",
                "50-100",
                "100-250",
                "250-500",
                "500-1000",
                "1000+"
            ]
        )

    if all(
        column in df.columns
        for column in [
            "lat",
            "long",
            "merch_lat",
            "merch_long"
        ]
    ):

        df["distance_km"] = np.sqrt(
            (df["lat"] - df["merch_lat"]) ** 2 +
            (df["long"] - df["merch_long"]) ** 2
        )

    else:

        df["distance_km"] = 0.0


# ============================================================
# DATA METRICS
# ============================================================

if analytics is not None:

    metrics = analytics["metrics"]

    total_transactions = metrics["total_transactions"]
    fraud_count = metrics["fraud_count"]
    legitimate_count = metrics["legitimate_count"]
    fraud_rate = metrics["fraud_rate"]

else:

    if (
        df is not None
        and "is_fraud" in df.columns
    ):

        total_transactions = len(df)

        fraud_count = int(
            df["is_fraud"].sum()
        )

        legitimate_count = (
            total_transactions - fraud_count
        )

        fraud_rate = (
            fraud_count / total_transactions * 100
            if total_transactions > 0
            else 0
        )

    else:

        total_transactions = 0
        fraud_count = 0
        legitimate_count = 0
        fraud_rate = 0

# ============================================================
# FEATURE ENGINEERING
# ============================================================

if df is not None:

    if "trans_date_trans_time" in df.columns:

        df["trans_date_trans_time"] = pd.to_datetime(
            df["trans_date_trans_time"],
            errors="coerce"
        )

        df["hour"] = (
            df["trans_date_trans_time"]
            .dt.hour
            .fillna(0)
            .astype(int)
        )

        df["month"] = (
            df["trans_date_trans_time"]
            .dt.month
            .fillna(0)
            .astype(int)
        )

        df["day_of_week"] = (
            df["trans_date_trans_time"]
            .dt.dayofweek
            .fillna(0)
            .astype(int)
        )

        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype(int)

        df["is_night"] = (
            (df["hour"] < 6) |
            (df["hour"] >= 22)
        ).astype(int)


    if "amt" in df.columns:

        df["amount_range"] = pd.cut(
            df["amt"],
            bins=[
                0,
                25,
                50,
                100,
                250,
                500,
                1000,
                np.inf
            ],
            labels=[
                "0-25",
                "25-50",
                "50-100",
                "100-250",
                "250-500",
                "500-1000",
                "1000+"
            ]
        )


    if all(
        column in df.columns
        for column in [
            "lat",
            "long",
            "merch_lat",
            "merch_long"
        ]
    ):

        df["distance_km"] = np.sqrt(
            (df["lat"] - df["merch_lat"]) ** 2 +
            (df["long"] - df["merch_long"]) ** 2
        )

    else:

        df["distance_km"] = 0.0


# ============================================================
# DATA METRICS
# ============================================================

if (
    df is not None
    and "is_fraud" in df.columns
):

    total_transactions = len(df)

    fraud_count = int(
        df["is_fraud"].sum()
    )

    legitimate_count = (
        total_transactions - fraud_count
    )

    fraud_rate = (
        fraud_count / total_transactions * 100
        if total_transactions > 0
        else 0
    )

else:

    total_transactions = 0
    fraud_count = 0
    legitimate_count = 0
    fraud_rate = 0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ FraudGuard AI")

    st.caption(
        "AI-Powered Financial Security Platform"
    )

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "🏠 Dashboard",
            "📊 Analytics",
            "🔮 Fraud Prediction",
            "🤖 Model Performance"
        ]
    )

    st.divider()

    if model_loaded:

        st.success(
            "🟢 Model Online"
        )

        st.caption(
            "Random Forest • ML Engine"
        )

    else:

        st.error(
            "🔴 Model Offline"
        )

    st.divider()

    if df is not None:

        st.info("📂 Full Dataset Loaded")

    elif analytics is not None:

        st.success("📊 Analytics Summary Loaded")

    else:

        st.warning("📂 Analytics Data Unavailable")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🛡️ FraudGuard AI")

    st.subheader(
        "Intelligent Credit Card Fraud Detection"
    )

    st.write(
        "Detect suspicious financial transactions "
        "using machine learning and interactive analytics."
    )

    st.divider()

    # ========================================================
    # DATASET AVAILABLE
    # ========================================================

    if df is not None or analytics is not None:

        st.subheader(
            "📈 Transaction Overview"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "💳 Total Transactions",
                f"{total_transactions:,}"
            )

        with c2:

            st.metric(
                "🚨 Fraud Transactions",
                f"{fraud_count:,}"
            )

        with c3:

            st.metric(
                "✅ Legitimate",
                f"{legitimate_count:,}"
            )

        with c4:

            st.metric(
                "⚠️ Fraud Rate",
                f"{fraud_rate:.2f}%"
            )

        st.divider()

        # ====================================================
        # CHARTS
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "🥧 Transaction Distribution"
            )

            distribution = pd.DataFrame(
                {
                    "Type": [
                        "Legitimate",
                        "Fraud"
                    ],
                    "Count": [
                        legitimate_count,
                        fraud_count
                    ]
                }
            )

            fig = px.pie(
                distribution,
                names="Type",
                values="Count",
                hole=0.60
            )

            fig.update_layout(
                height=400,
                margin=dict(
                    t=20,
                    b=20,
                    l=20,
                    r=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            st.subheader(
                "💰 Transaction Amount"
            )

            if df is not None:

                fig = px.histogram(
                    df,
                    x="amt",
                    color="is_fraud",
                    nbins=60,
                    labels={
                        "amt": "Transaction Amount",
                        "is_fraud": "Fraud"
                    }
                )

            else:

                hist_df = pd.DataFrame(
                    analytics["amount_histogram"]
                )

                hist_df = hist_df.melt(
                    id_vars=["label"],
                    value_vars=[
                        "legitimate",
                        "fraud"
                    ],
                    var_name="type",
                    value_name="count"
                )

                fig = px.bar(
                    hist_df,
                    x="label",
                    y="count",
                    color="type",
                    barmode="group",
                    labels={
                        "label": "Transaction Amount",
                        "count": "Transactions",
                        "type": "Transaction Type"
                    }
                )

            fig.update_layout(
                height=400,
                margin=dict(
                    t=20,
                    b=20,
                    l=20,
                    r=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        # ====================================================
        # QUICK INSIGHTS
        # ====================================================

        st.subheader(
            "🔎 Quick Insights"
        )

        if df is not None:

            fraud_avg = df[
                df["is_fraud"] == 1
            ]["amt"].mean()

            legitimate_avg = df[
                df["is_fraud"] == 0
            ]["amt"].mean()

        else:

            fraud_avg = analytics["metrics"][
                "average_fraud_amount"
            ]

            legitimate_avg = analytics["metrics"][
                "average_legitimate_amount"
            ]
        i1, i2, i3 = st.columns(3)

        with i1:

            st.metric(
                "💰 Average Fraud Amount",
                f"${fraud_avg:.2f}"
            )

        with i2:

            st.metric(
                "💵 Average Legitimate Amount",
                f"${legitimate_avg:.2f}"
            )

        with i3:

            st.metric(
                "🤖 Detection Model",
                "Random Forest"
            )

    # ========================================================
    # DATASET NOT AVAILABLE
    # ========================================================

    else:

        st.warning(
            "📂 Dashboard analytics are currently unavailable "
            "because fraudTest.csv is not included in the "
            "GitHub deployment."
        )

        st.info(
            "💡 Prediction and Model Performance continue "
            "to work without the large dataset."
        )

        st.subheader(
            "🤖 Fraud Detection Engine"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "🤖 ML Model",
                "Random Forest"
            )

        with c2:

            st.metric(
                "🟢 System",
                "Online" if model_loaded else "Offline"
            )

        with c3:

            st.metric(
                "🔮 Prediction",
                "Available"
            )

        st.divider()

        st.subheader(
            "🚀 Available Capabilities"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                "🔮 Real-time fraud prediction"
            )

            st.write(
                "🤖 Random Forest classification"
            )

            st.write(
                "📊 Model evaluation"
            )

        with col2:

            st.write(
                "🌲 Feature importance"
            )

            st.write(
                "📈 Interactive analytics"
            )

            st.write(
                "🛡️ Risk probability scoring"
            )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Fraud Analytics")

    st.write(
        "Explore transaction patterns and fraudulent behaviour."
    )

    st.divider()

    if df is None and analytics is None:

        st.warning(
            "📂 Analytics require fraudTest.csv."
        )

        
    else:

        # ====================================================
        # FRAUD BY HOUR
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "🕐 Fraud by Transaction Hour"
            )

            if df is not None:

                hourly = (
                    df.groupby("hour")["is_fraud"]
                    .sum()
                    .reset_index()
                )

            else:

                hourly = pd.DataFrame(
                    analytics["fraud_by_hour"]
                )

                hourly = hourly[
                    ["label", "fraud"]
                ]

                hourly.columns = [
                    "hour",
                    "is_fraud"
                ]

                hourly["hour"] = pd.to_numeric(
                    hourly["hour"],
                    errors="coerce"
                )

            fig = px.line(
                hourly,
                x="hour",
                y="is_fraud",
                markers=True,
                labels={
                    "hour": "Hour",
                    "is_fraud": "Fraud Transactions"
                }
            )

            fig.update_layout(
                height=380
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            st.subheader(
                "💰 Fraud by Amount Range"
            )

            if df is not None:

                amount_data = (
                    df[df["is_fraud"] == 1]
                    ["amount_range"]
                    .value_counts()
                    .sort_index()
                    .reset_index()
                )

                amount_data.columns = [
                    "Range",
                    "Fraud"
                ]

            else:

                amount_data = pd.DataFrame(
                    analytics["fraud_by_amount_range"]
                )

                amount_data = amount_data[
                    ["label", "fraud"]
                ]

                amount_data.columns = [
                    "Range",
                    "Fraud"
                ]

            fig = px.bar(
                amount_data,
                x="Range",
                y="Fraud"
            )

            fig.update_layout(
                height=380
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        # ====================================================
        # CATEGORY / GENDER
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "🛒 Fraud by Category"
            )

            if "category" in df.columns:

                if df is not None:

                    category = (
                        df[df["is_fraud"] == 1]
                        ["category"]
                        .value_counts()
                        .head(10)
                        .reset_index()
                    )

                    category.columns = [
                        "Category",
                        "Fraud"
                    ]

                else:

                    category = pd.DataFrame(
                        analytics["fraud_by_category"]
                    )

                    category = (
                        category
                        .sort_values("fraud", ascending=False)
                        .head(10)
                    )

                    category = category[
                        ["label", "fraud"]
                    ]

                    category.columns = [
                        "Category",
                        "Fraud"
                    ]

                category.columns = [
                    "Category",
                    "Fraud"
                ]

                fig = px.bar(
                    category,
                    x="Fraud",
                    y="Category",
                    orientation="h"
                )

                fig.update_layout(
                    height=420
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.info(
                    "Category information unavailable."
                )

        with col2:

            st.subheader(
                "👥 Fraud by Gender"
            )

            if "gender" in df.columns:

                if df is not None:

                    gender = (
                        df[df["is_fraud"] == 1]
                        ["gender"]
                        .value_counts()
                        .reset_index()
                    )

                    gender.columns = [
                        "Gender",
                        "Fraud"
                    ]

                else:

                    gender = pd.DataFrame(
                        analytics["fraud_by_gender"]
                    )

                    gender = gender[
                        ["label", "fraud"]
                    ]

                    gender.columns = [
                        "Gender",
                        "Fraud"
                    ]

                fig = px.pie(
                    gender,
                    names="Gender",
                    values="Fraud",
                    hole=0.55
                )

                fig.update_layout(
                    height=420
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.info(
                    "Gender information unavailable."
                )

        st.divider()

        # ====================================================
        # DAY / MONTH
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "📅 Fraud by Day of Week"
            )

            if df is not None:

                day = (
                    df.groupby("day_of_week")["is_fraud"]
                    .sum()
                    .reset_index()
                )

            else:

                day = pd.DataFrame(
                    analytics["fraud_by_day"]
                )

                day = day[
                    ["label", "fraud"]
                ]

                day.columns = [
                    "day_of_week",
                    "is_fraud"
                ]

            fig = px.bar(
                day,
                x="day_of_week",
                y="is_fraud",
                labels={
                    "day_of_week": "Day of Week",
                    "is_fraud": "Fraud Transactions"
                }
            )

            fig.update_layout(
                height=380
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            st.subheader(
                "📆 Fraud by Month"
            )

            if df is not None:

                month = (
                    df.groupby("month")["is_fraud"]
                    .sum()
                    .reset_index()
                )

            else:

                month = pd.DataFrame(
                    analytics["fraud_by_month"]
                )

                month = month[
                    ["label", "fraud"]
                ]

                month.columns = [
                    "month",
                    "is_fraud"
                ]

                month["month"] = pd.to_numeric(
                    month["month"],
                    errors="coerce"
                )

            fig = px.line(
                month,
                x="month",
                y="is_fraud",
                markers=True,
                labels={
                    "month": "Month",
                    "is_fraud": "Fraud Transactions"
                }
            )

            fig.update_layout(
                height=380
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        # ====================================================
        # WEEKEND / NIGHT
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "🗓️ Weekend vs Weekday"
            )

            if df is not None:

                weekend = (
                    df.groupby("is_weekend")["is_fraud"]
                    .sum()
                    .reset_index()
                )

            else:

                weekend = pd.DataFrame(
                    analytics["fraud_by_weekend"]
                )

                weekend = weekend[
                    ["label", "fraud"]
                ]

                weekend.columns = [
                    "is_weekend",
                    "is_fraud"
                ]

            weekend["is_weekend"] = (
                weekend["is_weekend"]
                .map({
                    0: "Weekday",
                    1: "Weekend"
                })
            )

            fig = px.bar(
                weekend,
                x="is_weekend",
                y="is_fraud",
                labels={
                    "is_weekend": "Transaction Day",
                    "is_fraud": "Fraud"
                }
            )

            fig.update_layout(
                height=380
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            st.subheader(
                "🌙 Night vs Day Fraud"
            )

            if df is not None:

                night = (
                    df.groupby("is_night")["is_fraud"]
                    .sum()
                    .reset_index()
                )

            else:

                night = pd.DataFrame(
                    analytics["fraud_by_night"]
                )

                night = night[
                    ["label", "fraud"]
                ]

                night.columns = [
                    "is_night",
                    "is_fraud"
                ]

            night["is_night"] = (
                night["is_night"]
                .map({
                    0: "Day",
                    1: "Night"
                })
            )

            fig = px.pie(
                night,
                names="is_night",
                values="is_fraud",
                hole=0.55
            )

            fig.update_layout(
                height=380
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        # ====================================================
        # DISTANCE
        # ====================================================

        st.subheader(
            "📍 Transaction Distance Analysis"
        )

        if df is not None:

            distance_data = (
                df.groupby("is_fraud")["distance_km"]
                .mean()
                .reset_index()
            )

        else:

            distance_data = pd.DataFrame(
                analytics["distance_analysis"]
            )

            distance_data = distance_data[
                ["is_fraud", "mean"]
            ]

            distance_data.columns = [
                "is_fraud",
                "distance_km"
            ]

        distance_data["is_fraud"] = (
            distance_data["is_fraud"]
            .map({
                0: "Legitimate",
                1: "Fraud"
            })
        )

        fig = px.bar(
            distance_data,
            x="is_fraud",
            y="distance_km",
            labels={
                "is_fraud": "Transaction Type",
                "distance_km": "Average Distance"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        # ====================================================
        # CORRELATION
        # ====================================================

        st.subheader(
            "🔗 Feature Correlation"
        )

        correlation = None

        if df is not None:

            numeric = df.select_dtypes(
                include=np.number
            )

            if "is_fraud" in numeric.columns:

                correlation = (
                    numeric.corr()["is_fraud"]
                    .drop("is_fraud")
                    .sort_values()
                )

        else:

            corr_df = pd.DataFrame(
                analytics["correlation"]
            )

            if "is_fraud" in corr_df.columns:

                correlation = (
                    corr_df["is_fraud"]
                    .drop("is_fraud")
                    .sort_values()
                )


        if correlation is not None:

            fig = px.bar(
                correlation,
                orientation="h",
                labels={
                    "value": "Correlation",
                    "index": "Feature"
                }
            )

            fig.update_layout(
                height=520
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "⚠️ Correlation data unavailable."
            )

# ============================================================
# FRAUD PREDICTION
# ============================================================

elif page == "🔮 Fraud Prediction":

    st.title("🔮 Fraud Risk Scanner")

    st.write(
        "Evaluate a transaction using the trained "
        "Random Forest machine learning model."
    )

    st.divider()

    if not model_loaded:

        st.error(
            "❌ Model is not available."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "💳 Transaction Information"
            )

            amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=100.0,
                step=1.0
            )

            hour = st.slider(
                "Transaction Hour",
                min_value=0,
                max_value=23,
                value=12
            )

            distance = st.number_input(
                "Distance from Merchant (km)",
                min_value=0.0,
                value=10.0,
                step=1.0
            )

        with col2:

            st.subheader(
                "👤 Customer Information"
            )

            age = st.number_input(
                "Customer Age",
                min_value=18,
                max_value=100,
                value=30
            )

            city_population = st.number_input(
                "City Population",
                min_value=0,
                value=10000,
                step=1000
            )

            weekend = st.selectbox(
                "Weekend Transaction",
                [
                    "No",
                    "Yes"
                ]
            )

        st.divider()

        scan = st.button(
            "🛡️ SCAN TRANSACTION",
            use_container_width=True,
            type="primary"
        )

        if scan:

            try:

                features = model.feature_names_in_

                data = pd.DataFrame(
                    0,
                    index=[0],
                    columns=features
                )

                feature_values = {
                    "amt": amount,
                    "hour": hour,
                    "age": age,
                    "distance_km": distance,
                    "city_pop": city_population,
                    "is_weekend": (
                        1
                        if weekend == "Yes"
                        else 0
                    ),
                    "unix_time": hour * 3600,
                    "is_night": int(
                        hour < 6 or hour >= 22
                    ),
                    "large_distance": int(
                        distance > 100
                    ),
                    "is_high_amount": int(
                        amount > 500
                    )
                }

                for feature, value in feature_values.items():

                    if feature in data.columns:

                        data[feature] = value

                prediction = model.predict(
                    data
                )[0]

                probability = model.predict_proba(
                    data
                )[0][1]

                st.divider()

                if prediction == 1:

                    st.error(
                        "🚨 FRAUDULENT TRANSACTION DETECTED"
                    )

                    st.warning(
                        "High-risk transaction pattern identified."
                    )

                    st.metric(
                        "🚨 Fraud Risk Score",
                        f"{probability * 100:.2f}%"
                    )

                else:

                    st.success(
                        "✅ TRANSACTION APPEARS LEGITIMATE"
                    )

                    st.info(
                        "No strong fraudulent pattern detected."
                    )

                    st.metric(
                        "🛡️ Fraud Probability",
                        f"{probability * 100:.2f}%"
                    )

                st.subheader(
                    "📊 Risk Probability"
                )

                st.progress(
                    min(
                        max(
                            float(probability),
                            0.0
                        ),
                        1.0
                    )
                )

                if probability >= 0.75:

                    st.error(
                        "Risk Level: HIGH"
                    )

                elif probability >= 0.40:

                    st.warning(
                        "Risk Level: MEDIUM"
                    )

                else:

                    st.success(
                        "Risk Level: LOW"
                    )

            except Exception as e:

                st.error(
                    "❌ Prediction could not be completed."
                )

                st.exception(e)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    st.write(
        "Evaluation metrics and machine learning insights."
    )

    st.divider()

    # ========================================================
    # METRICS
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "🎯 Accuracy",
            "99.83%"
        )

    with c2:

        st.metric(
            "🔎 Precision",
            "95.00%"
        )

    with c3:

        st.metric(
            "📡 Recall",
            "58.00%"
        )

    with c4:

        st.metric(
            "🏆 ROC-AUC",
            "96.95%"
        )

    st.divider()

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.subheader(
        "📋 Classification Report"
    )

    report = pd.DataFrame(
        {
            "Class": [
                "Legitimate",
                "Fraud"
            ],
            "Precision": [
                1.00,
                0.95
            ],
            "Recall": [
                1.00,
                0.58
            ],
            "F1-Score": [
                1.00,
                0.72
            ]
        }
    )

    st.dataframe(
        report,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "🔢 Confusion Matrix"
    )

    cm = np.array(
        [
            [110715, 0],
            [180, 249]
        ]
    )

    fig = px.imshow(
        cm,
        text_auto=True,
        x=[
            "Predicted Legitimate",
            "Predicted Fraud"
        ],
        y=[
            "Actual Legitimate",
            "Actual Fraud"
        ],
        labels={
            "x": "Prediction",
            "y": "Actual",
            "color": "Count"
        }
    )

    fig.update_layout(
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.subheader(
        "🌲 Feature Importance"
    )

    if (
        model_loaded
        and hasattr(
            model,
            "feature_importances_"
        )
        and hasattr(
            model,
            "feature_names_in_"
        )
    ):

        importance = pd.DataFrame(
            {
                "Feature":
                    model.feature_names_in_,

                "Importance":
                    model.feature_importances_
            }
        )

        importance = (
            importance
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(15)
        )

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h"
        )

        fig.update_layout(
            height=550,
            yaxis=dict(
                categoryorder="total ascending"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Feature importance is unavailable."
        )

    st.divider()

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.subheader(
        "🌲 Random Forest Classifier"
    )

    st.write(
        "FraudGuard AI uses a Random Forest classifier "
        "to identify potentially fraudulent transactions. "
        "Multiple decision trees work together to improve "
        "prediction reliability and reduce overfitting."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Model Type",
            "Random Forest"
        )

    with c2:

        st.metric(
            "ML Engine",
            "Scikit-Learn"
        )

    with c3:

        st.metric(
            "Deployment",
            "Streamlit"
        )

    st.info(
        "ℹ️ fraudTest.csv is not required for "
        "Fraud Prediction or Model Performance."
    )