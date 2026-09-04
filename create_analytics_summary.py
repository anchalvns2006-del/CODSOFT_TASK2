import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("notebooks/fraudTest.csv")
OUTPUT_PATH = Path("analytics_summary.json")


print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# -----------------------------
# Basic cleaning
# -----------------------------
df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"],
    errors="coerce"
)

df["hour"] = df["trans_date_trans_time"].dt.hour
df["month"] = df["trans_date_trans_time"].dt.month
df["day_of_week"] = df["trans_date_trans_time"].dt.day_name()

df["is_weekend"] = (
    df["trans_date_trans_time"].dt.dayofweek >= 5
).astype(int)

df["is_night"] = (
    (df["hour"] >= 22) | (df["hour"] < 6)
).astype(int)

# Distance feature
if "lat" in df.columns and "merch_lat" in df.columns:
    lat1 = np.radians(df["lat"])
    lat2 = np.radians(df["merch_lat"])
    lon1 = np.radians(df["long"])
    lon2 = np.radians(df["merch_long"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    df["distance_km"] = (
        6371 * 2 * np.arcsin(np.sqrt(a))
    )
else:
    df["distance_km"] = 0


# -----------------------------
# Amount range
# -----------------------------
df["amount_range"] = pd.cut(
    df["amt"],
    bins=[-np.inf, 25, 50, 100, 250, 500, 1000, np.inf],
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


# -----------------------------
# Basic metrics
# -----------------------------
total_transactions = int(len(df))
fraud_count = int(df["is_fraud"].sum())
legitimate_count = int(total_transactions - fraud_count)

fraud_rate = (
    fraud_count / total_transactions * 100
    if total_transactions > 0
    else 0
)

avg_amount = float(df["amt"].mean())
avg_fraud_amount = float(
    df.loc[df["is_fraud"] == 1, "amt"].mean()
) if fraud_count else 0

avg_legitimate_amount = float(
    df.loc[df["is_fraud"] == 0, "amt"].mean()
) if legitimate_count else 0


# -----------------------------
# Helper function
# -----------------------------
def grouped_fraud_counts(column):
    result = (
        df.groupby(column, observed=False)["is_fraud"]
        .agg(["count", "sum"])
        .reset_index()
    )

    result.columns = [
        "label",
        "total",
        "fraud"
    ]

    result["legitimate"] = (
        result["total"] - result["fraud"]
    )

    result["label"] = result["label"].astype(str)

    return result.to_dict(orient="records")


# -----------------------------
# Histogram data
# -----------------------------
bins = [
    0,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
    float("inf")
]

labels = [
    "0-10",
    "10-25",
    "25-50",
    "50-100",
    "100-250",
    "250-500",
    "500-1000",
    "1000-2500",
    "2500-5000",
    "5000-10000",
    "10000+"
]

df["amount_bin"] = pd.cut(
    df["amt"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

amount_hist = (
    df.groupby(
        ["amount_bin", "is_fraud"],
        observed=False
    )
    .size()
    .reset_index(name="count")
)

histogram = []

for label in labels:
    legitimate = amount_hist[
        (amount_hist["amount_bin"].astype(str) == label)
        & (amount_hist["is_fraud"] == 0)
    ]["count"]

    fraud = amount_hist[
        (amount_hist["amount_bin"].astype(str) == label)
        & (amount_hist["is_fraud"] == 1)
    ]["count"]

    histogram.append({
        "label": label,
        "legitimate": int(legitimate.iloc[0]) if len(legitimate) else 0,
        "fraud": int(fraud.iloc[0]) if len(fraud) else 0
    })


# -----------------------------
# Correlation data
# -----------------------------
correlation_columns = [
    "amt",
    "hour",
    "age",
    "city_pop",
    "distance_km",
    "is_weekend",
    "is_night",
    "is_fraud"
]

available_columns = [
    col for col in correlation_columns
    if col in df.columns
]

correlation = (
    df[available_columns]
    .corr()
    .round(4)
    .fillna(0)
    .to_dict()
)


# -----------------------------
# Distance analysis
# -----------------------------
distance_analysis = (
    df.groupby("is_fraud")["distance_km"]
    .agg(["count", "mean", "median", "max"])
    .reset_index()
)

distance_analysis["is_fraud"] = (
    distance_analysis["is_fraud"]
    .astype(int)
)

distance_analysis = distance_analysis.round(2).to_dict(
    orient="records"
)


# -----------------------------
# Final summary
# -----------------------------
summary = {
    "metrics": {
        "total_transactions": total_transactions,
        "fraud_count": fraud_count,
        "legitimate_count": legitimate_count,
        "fraud_rate": round(fraud_rate, 4),
        "average_amount": round(avg_amount, 2),
        "average_fraud_amount": round(avg_fraud_amount, 2),
        "average_legitimate_amount": round(
            avg_legitimate_amount, 2
        )
    },

    "fraud_by_hour": grouped_fraud_counts("hour"),

    "fraud_by_amount_range": grouped_fraud_counts(
        "amount_range"
    ),

    "fraud_by_category": grouped_fraud_counts(
        "category"
    ) if "category" in df.columns else [],

    "fraud_by_gender": grouped_fraud_counts(
        "gender"
    ) if "gender" in df.columns else [],

    "fraud_by_day": grouped_fraud_counts(
        "day_of_week"
    ),

    "fraud_by_month": grouped_fraud_counts(
        "month"
    ),

    "fraud_by_weekend": grouped_fraud_counts(
        "is_weekend"
    ),

    "fraud_by_night": grouped_fraud_counts(
        "is_night"
    ),

    "amount_histogram": histogram,

    "correlation": correlation,

    "distance_analysis": distance_analysis
}


# -----------------------------
# Save JSON
# -----------------------------
with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        summary,
        file,
        indent=2,
        allow_nan=False
    )


file_size_mb = OUTPUT_PATH.stat().st_size / (
    1024 * 1024
)

print()
print("====================================")
print("Analytics summary created successfully!")
print("====================================")
print(f"Output: {OUTPUT_PATH}")
print(f"Size: {file_size_mb:.2f} MB")
print(f"Transactions: {total_transactions:,}")
print(f"Fraud transactions: {fraud_count:,}")
print(f"Fraud rate: {fraud_rate:.2f}%")
print("====================================")