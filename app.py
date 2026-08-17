import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================

css_file = Path("style.css")

if css_file.exists():
    st.markdown(
        f"<style>{css_file.read_text()}</style>",
        unsafe_allow_html=True
    )

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = Path("notebooks/fraud_detection_model.pkl")
DATA_PATH = Path("notebooks/fraudTest.csv")

model = joblib.load(MODEL_PATH)

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
else:
    st.error("⚠️ Dataset not found. Check fraudTest.csv path.")
    st.stop()


# ============================================================
# FEATURE ENGINEERING FOR ANALYTICS
# ============================================================

df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"]
)

df["hour"] = df["trans_date_trans_time"].dt.hour
df["month"] = df["trans_date_trans_time"].dt.month
df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

df["is_night"] = (
    (df["hour"] < 6) |
    (df["hour"] >= 22)
).astype(int)

df["amount_range"] = pd.cut(
    df["amt"],
    bins=[0, 25, 50, 100, 250, 500, 1000, np.inf],
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

df["distance_km"] = np.sqrt(
    (df["lat"] - df["merch_lat"]) ** 2 +
    (df["long"] - df["merch_long"]) ** 2
)
# ============================================================
# BASIC DATA
# ============================================================

total_transactions = len(df)
fraud_count = int(df["is_fraud"].sum())
legitimate_count = total_transactions - fraud_count
fraud_rate = fraud_count / total_transactions * 100

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div class="brand">
        <div class="brand-icon">🛡️</div>
        <div>
            <h2>FraudGuard</h2>
            <span>AI Security Platform</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "🏠 Dashboard",
        "📊 Analytics",
        "🔮 Fraud Prediction",
        "🤖 Model Performance"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div class="model-status">
        <span class="online-dot"></span>
        <b>Model Online</b>
        <small>Random Forest • ML Engine</small>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="hero">
            <div>
                <div class="eyebrow">AI-POWERED FINANCIAL SECURITY</div>
                <h1>FraudGuard AI</h1>
                <p>
                    Intelligent credit card fraud detection
                    powered by machine learning.
                </p>
            </div>
            <div class="hero-icon">🛡️</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## 📈 Transaction Overview")

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        ("💳", "Total Transactions", f"{total_transactions:,}", "All transactions"),
        ("🚨", "Fraud Transactions", f"{fraud_count:,}", "Detected fraud"),
        ("✅", "Legitimate", f"{legitimate_count:,}", "Safe transactions"),
        ("⚠️", "Fraud Rate", f"{fraud_rate:.2f}%", "Overall fraud rate")
    ]

    for col, (icon, title, value, sub) in zip(
        [c1, c2, c3, c4],
        cards
    ):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-title">{title}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # FRAUD VS LEGITIMATE
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🥧 Transaction Distribution")

        dist = pd.DataFrame({
            "Type": ["Legitimate", "Fraud"],
            "Count": [legitimate_count, fraud_count]
        })

        fig = px.pie(
            dist,
            names="Type",
            values="Count",
            hole=0.62
        )

        fig.update_layout(
            showlegend=True,
            margin=dict(t=20, b=20, l=10, r=10),
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------

    with col2:

        st.markdown("### 💰 Transaction Amount")

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

        fig.update_layout(
            height=350,
            margin=dict(t=20, b=20, l=10, r=10)
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # QUICK INSIGHTS
    # --------------------------------------------------------

    st.markdown("## 🔎 Quick Insights")

    fraud_avg = df[df["is_fraud"] == 1]["amt"].mean()
    normal_avg = df[df["is_fraud"] == 0]["amt"].mean()

    i1, i2, i3 = st.columns(3)

    with i1:
        st.markdown(
            f"""
            <div class="insight-card">
                <span>💰</span>
                <b>Average Fraud Amount</b>
                <strong>${fraud_avg:.2f}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i2:
        st.markdown(
            f"""
            <div class="insight-card">
                <span>💵</span>
                <b>Average Legitimate Amount</b>
                <strong>${normal_avg:.2f}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i3:
        st.markdown(
            """
            <div class="insight-card">
                <span>🤖</span>
                <b>Detection Model</b>
                <strong>Random Forest</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.markdown(
        """
        <div class="page-header">
            <div class="eyebrow">DATA INTELLIGENCE</div>
            <h1>Fraud Analytics</h1>
            <p>Explore transaction patterns and fraud behaviour.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # 1. FRAUD BY HOUR
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🕐 Fraud by Transaction Hour")

        hourly = (
            df.groupby("hour")["is_fraud"]
            .sum()
            .reset_index()
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

        fig.update_layout(height=350)

        st.plotly_chart(fig, use_container_width=True)

    # ========================================================

    with col2:

        st.markdown("### 💰 Fraud by Amount Range")

        bins = [
            0, 25, 50, 100,
            250, 500,
            1000, float("inf")
        ]

        labels = [
            "0-25",
            "25-50",
            "50-100",
            "100-250",
            "250-500",
            "500-1000",
            "1000+"
        ]

        temp = df.copy()

        temp["amount_range_chart"] = pd.cut(
            temp["amt"],
            bins=bins,
            labels=labels
        )

        amount_data = (
            temp[temp["is_fraud"] == 1]
            ["amount_range_chart"]
            .value_counts()
            .reindex(labels)
            .fillna(0)
            .reset_index()
        )

        amount_data.columns = ["Range", "Fraud"]

        fig = px.bar(
            amount_data,
            x="Range",
            y="Fraud"
        )

        fig.update_layout(height=350)

        st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # 2. CATEGORY
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🛒 Fraud by Category")

        category = (
            df[df["is_fraud"] == 1]
            ["category"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        category.columns = ["Category", "Fraud"]

        fig = px.bar(
            category,
            x="Fraud",
            y="Category",
            orientation="h"
        )

        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

    # ========================================================

    with col2:

        st.markdown("### 👥 Fraud by Gender")

        gender = (
            df[df["is_fraud"] == 1]
            ["gender"]
            .value_counts()
            .reset_index()
        )

        gender.columns = ["Gender", "Fraud"]

        fig = px.pie(
            gender,
            names="Gender",
            values="Fraud",
            hole=0.5
        )

        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # 3. DAY / MONTH
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📅 Fraud by Day of Week")

        if "day_of_week" in df.columns:

            day = (
                df.groupby("day_of_week")["is_fraud"]
                .sum()
                .reset_index()
            )

            fig = px.bar(
                day,
                x="day_of_week",
                y="is_fraud",
                labels={
                    "day_of_week": "Day",
                    "is_fraud": "Fraud"
                }
            )

            fig.update_layout(height=350)

            st.plotly_chart(fig, use_container_width=True)

    # ========================================================

    with col2:

        st.markdown("### 📆 Fraud by Month")

        if "month" in df.columns:

            month = (
                df.groupby("month")["is_fraud"]
                .sum()
                .reset_index()
            )

            fig = px.line(
                month,
                x="month",
                y="is_fraud",
                markers=True
            )

            fig.update_layout(height=350)

            st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # 4. WEEKEND / NIGHT
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🗓️ Weekend vs Weekday")

        weekend = (
            df.groupby("is_weekend")["is_fraud"]
            .sum()
            .reset_index()
        )

        weekend["is_weekend"] = weekend["is_weekend"].map({
            0: "Weekday",
            1: "Weekend"
        })

        fig = px.bar(
            weekend,
            x="is_weekend",
            y="is_fraud"
        )

        fig.update_layout(height=350)

        st.plotly_chart(fig, use_container_width=True)

    # ========================================================

    with col2:

        st.markdown("### 🌙 Night vs Day Fraud")

        night = (
            df.groupby("is_night")["is_fraud"]
            .sum()
            .reset_index()
        )

        night["is_night"] = night["is_night"].map({
            0: "Day",
            1: "Night"
        })

        fig = px.pie(
            night,
            names="is_night",
            values="is_fraud",
            hole=0.55
        )

        fig.update_layout(height=350)

        st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # 5. DISTANCE
    # ========================================================

    st.markdown("### 📍 Transaction Distance Analysis")

    if "distance_km" in df.columns:

        distance_data = (
            df.groupby("is_fraud")["distance_km"]
            .mean()
            .reset_index()
        )

        distance_data["is_fraud"] = distance_data["is_fraud"].map({
            0: "Legitimate",
            1: "Fraud"
        })

        fig = px.bar(
            distance_data,
            x="is_fraud",
            y="distance_km",
            labels={
                "is_fraud": "Transaction Type",
                "distance_km": "Average Distance (km)"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ========================================================
    # 6. CORRELATION
    # ========================================================

    st.markdown("### 🔗 Feature Correlation")

    numeric = df.select_dtypes(include=np.number)

    if "is_fraud" in numeric.columns:

        corr = (
            numeric.corr()["is_fraud"]
            .drop("is_fraud")
            .sort_values()
        )

        fig = px.bar(
            corr,
            orientation="h",
            labels={
                "value": "Correlation",
                "index": "Feature"
            }
        )

        fig.update_layout(height=500)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ============================================================
# PREDICTION
# ============================================================

elif page == "🔮 Fraud Prediction":

    st.markdown(
        """
        <div class="page-header">
            <div class="eyebrow">REAL-TIME AI ANALYSIS</div>
            <h1>Fraud Risk Scanner</h1>
            <p>Evaluate a transaction using the trained ML model.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 💳 Transaction Information")

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0
        )

        hour = st.slider(
            "Transaction Hour",
            0,
            23,
            12
        )

        distance = st.number_input(
            "Distance from Merchant (km)",
            min_value=0.0,
            value=10.0
        )

    with col2:

        st.markdown("### 👤 Customer Information")

        age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=30
        )

        city_population = st.number_input(
            "City Population",
            min_value=0,
            value=10000
        )

        weekend = st.selectbox(
            "Weekend Transaction",
            ["No", "Yes"]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "🛡️ SCAN TRANSACTION",
        use_container_width=True
    ):

        features = model.feature_names_in_

        data = pd.DataFrame(
            0,
            index=[0],
            columns=features
        )

        data["amt"] = amount
        data["hour"] = hour
        data["age"] = age
        data["distance_km"] = distance
        data["city_pop"] = city_population

        data["is_weekend"] = (
            1 if weekend == "Yes" else 0
        )

        data["unix_time"] = hour * 3600

        data["is_night"] = int(
            hour < 6 or hour >= 22
        )

        data["large_distance"] = int(
            distance > 100
        )

        data["is_high_amount"] = int(
            amount > 500
        )

        prediction = model.predict(data)[0]

        probability = model.predict_proba(data)[0][1]

        st.markdown("---")

        if prediction == 1:

            st.markdown(
                f"""
                <div class="risk-danger">
                    <div class="big-icon">🚨</div>
                    <h2>Fraudulent Transaction Detected</h2>
                    <p>High-risk transaction pattern identified.</p>
                    <div class="risk-score">
                        Risk Score: {probability * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="risk-safe">
                    <div class="big-icon">✓</div>
                    <h2>Transaction Appears Legitimate</h2>
                    <p>No strong fraudulent pattern detected.</p>
                    <div class="risk-score">
                        Fraud Probability: {probability * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### 📊 Risk Probability")

        st.progress(
            min(float(probability), 1.0)
        )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.markdown(
        """
        <div class="page-header">
            <div class="eyebrow">MACHINE LEARNING ENGINE</div>
            <h1>Model Performance</h1>
            <p>Evaluation metrics and model insights.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    performance = [
        ("🎯", "Accuracy", "99.83%"),
        ("🔎", "Precision", "95.00%"),
        ("📡", "Recall", "58.00%"),
        ("🏆", "ROC-AUC", "96.95%")
    ]

    for col, (icon, title, value) in zip(
        [c1, c2, c3, c4],
        performance
    ):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-title">{title}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-sub">Random Forest</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.markdown("### 📋 Classification Report")

    report = pd.DataFrame({
        "Class": [
            "Legitimate",
            "Fraud"
        ],
        "Precision": [
            "1.00",
            "0.95"
        ],
        "Recall": [
            "1.00",
            "0.58"
        ],
        "F1-Score": [
            "1.00",
            "0.72"
        ]
    })

    st.dataframe(
        report,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.markdown("### 🔢 Confusion Matrix")

    cm = np.array([
        [110715, 0],
        [180, 249]
    ])

    fig = px.imshow(
        cm,
        text_auto=True,
        x=["Predicted Legitimate", "Predicted Fraud"],
        y=["Actual Legitimate", "Actual Fraud"],
        labels=dict(
            x="Prediction",
            y="Actual",
            color="Count"
        )
    )

    fig.update_layout(height=450)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.markdown("### 🌲 Feature Importance")

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({
            "Feature": model.feature_names_in_,
            "Importance": model.feature_importances_
        })

        importance = (
            importance
            .sort_values("Importance", ascending=False)
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

    # ========================================================
    # MODEL INFO
    # ========================================================

    st.markdown(
        """
        <div class="model-panel">
            <div class="big-icon">🌲</div>
            <div>
                <h3>Random Forest Classifier</h3>
                <p>
                    FraudGuard uses a Random Forest model to identify
                    potentially fraudulent transactions. Multiple
                    decision trees work together to improve prediction
                    reliability and reduce overfitting.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )