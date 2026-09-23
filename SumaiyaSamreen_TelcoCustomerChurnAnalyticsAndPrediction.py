"""
Telco Customer Churn Analytics & Prediction
End-to-End Streamlit Application
Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from sklearn.inspection import permutation_importance
import joblib

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────
CHURN_COLORS = {"No": "#2196F3", "Yes": "#F44336"}
PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

# ─────────────────────────────────────────────
# DATA LOADING & VALIDATION
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df

@st.cache_data
def clean_and_preprocess(df_raw):
    df = df_raw.copy()

    # Fix TotalCharges: blank strings → NaN → impute with MonthlyCharges (tenure=0 customers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_tc = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    # Binary encode Churn
    df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

    # Encode SeniorCitizen as label
    df["SeniorCitizen_Label"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return df, missing_tc

@st.cache_data
def engineer_features(df):
    df = df.copy()

    # Tenure bands
    bins = [-1, 12, 24, 48, 72]
    labels = ["0–12 mo", "13–24 mo", "25–48 mo", "49–72 mo"]
    df["Tenure_Band"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    # Monthly charge bands
    df["Charge_Band"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 65, 95, 200],
        labels=["Low (<35)", "Mid (35–65)", "High (65–95)", "Very High (>95)"],
    )

    # Number of add-on services
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["Num_Services"] = df[service_cols].apply(
        lambda row: sum(v == "Yes" for v in row), axis=1
    )

    # Avg monthly charge per year of tenure
    df["Charge_Per_Month"] = df["TotalCharges"] / (df["tenure"] + 1)

    return df

@st.cache_data
def prepare_model_data(df):
    """Full preprocessing pipeline → feature matrix + target."""
    drop_cols = ["customerID", "Churn", "Churn_Binary",
                 "SeniorCitizen_Label", "Tenure_Band", "Charge_Band", "Charge_Per_Month",
                 "Num_Services"]
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode categoricals
    cat_cols = feature_df.select_dtypes(include="object").columns.tolist()
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col].astype(str))
        le_dict[col] = le

    X = feature_df.values
    y = df["Churn_Binary"].values
    feature_names = feature_df.columns.tolist()

    return X, y, feature_names, le_dict, feature_df

@st.cache_data
def train_models(df):
    X, y, feature_names, le_dict, feature_df = prepare_model_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        if name in ("Logistic Regression", "K-Nearest Neighbors"):
            Xtr, Xte = X_train_sc, X_test_sc
        else:
            Xtr, Xte = X_train, X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "y_pred": y_pred,
            "y_prob": y_prob,
            "cm": confusion_matrix(y_test, y_pred),
        }
        trained[name] = (model, Xtr is X_train_sc)  # flag: uses scaled data

    return results, trained, X_train, X_test, y_train, y_test, scaler, feature_names, le_dict

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.title("📡 Telco Churn")
st.sidebar.markdown("---")
pages = [
    "🏠 Overview & KPI Dashboard",
    "🔍 Data Explorer",
    "📊 Exploratory Data Analysis",
    "💡 Business Insights",
    "⚙️ Feature Engineering",
    "🤖 Model Training & Evaluation",
    "📈 Model Comparison",
    "🔮 Customer Prediction",
]
selected_page = st.sidebar.radio("Navigate", pages)
st.sidebar.markdown("---")
st.sidebar.info("Dataset: Telco Customer Churn (Kaggle)\n7,043 customers · 21 features")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df_raw = load_data()
df, missing_tc = clean_and_preprocess(df_raw)
df = engineer_features(df)

# ─────────────────────────────────────────────
# PAGE 1 — OVERVIEW & KPI DASHBOARD
# ─────────────────────────────────────────────
if selected_page == "🏠 Overview & KPI Dashboard":
    st.title("📡 Telco Customer Churn Analytics & Prediction")
    st.markdown(
        "A complete end-to-end analytics and machine-learning project exploring "
        "customer churn patterns and building predictive models."
    )

    churn_rate = df["Churn_Binary"].mean() * 100
    total_customers = len(df)
    churned = df["Churn_Binary"].sum()
    avg_tenure = df["tenure"].mean()
    avg_monthly = df["MonthlyCharges"].mean()
    revenue_at_risk = df[df["Churn"] == "Yes"]["MonthlyCharges"].sum()

    st.markdown("### 📌 Key Performance Indicators")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Churn Rate", f"{churn_rate:.1f}%")
    c4.metric("Avg Tenure (mo)", f"{avg_tenure:.1f}")
    c5.metric("Avg Monthly Charge", f"${avg_monthly:.2f}")
    c6.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    # Churn distribution pie
    with col1:
        st.markdown("#### Churn Distribution")
        fig = px.pie(
            values=[total_customers - churned, churned],
            names=["Retained", "Churned"],
            color_discrete_sequence=["#2196F3", "#F44336"],
            hole=0.45,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Churn by Contract
    with col2:
        st.markdown("#### Churn Rate by Contract Type")
        ct = df.groupby("Contract")["Churn_Binary"].mean().reset_index()
        ct.columns = ["Contract", "Churn Rate"]
        ct["Churn Rate %"] = ct["Churn Rate"] * 100
        fig2 = px.bar(
            ct, x="Contract", y="Churn Rate %",
            color="Churn Rate %", color_continuous_scale="RdYlGn_r",
            text=ct["Churn Rate %"].map("{:.1f}%".format),
        )
        fig2.update_layout(height=300, margin=dict(t=10, b=10), coloraxis_showscale=False)
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    # Churn by Internet Service
    with col3:
        st.markdown("#### Churn Rate by Internet Service")
        isc = df.groupby("InternetService")["Churn_Binary"].mean().reset_index()
        isc.columns = ["InternetService", "Churn Rate"]
        isc["Churn Rate %"] = isc["Churn Rate"] * 100
        fig3 = px.bar(
            isc, x="InternetService", y="Churn Rate %",
            color="Churn Rate %", color_continuous_scale="RdYlGn_r",
            text=isc["Churn Rate %"].map("{:.1f}%".format),
        )
        fig3.update_layout(height=300, margin=dict(t=10, b=10), coloraxis_showscale=False)
        fig3.update_traces(textposition="outside")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    col4, col5 = st.columns(2)

    # Tenure distribution by churn
    with col4:
        st.markdown("#### Tenure Distribution by Churn")
        fig4 = px.histogram(
            df, x="tenure", color="Churn", barmode="overlay",
            color_discrete_map=CHURN_COLORS, nbins=40, opacity=0.7,
        )
        fig4.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)

    # Monthly Charges by churn
    with col5:
        st.markdown("#### Monthly Charges by Churn")
        fig5 = px.box(
            df, x="Churn", y="MonthlyCharges", color="Churn",
            color_discrete_map=CHURN_COLORS,
        )
        fig5.update_layout(height=320, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Churn Rate Across Tenure Bands")
    tb = df.groupby("Tenure_Band", observed=True)["Churn_Binary"].agg(["mean", "count"]).reset_index()
    tb.columns = ["Tenure Band", "Churn Rate", "Count"]
    tb["Churn Rate %"] = tb["Churn Rate"] * 100
    fig6 = px.bar(
        tb, x="Tenure Band", y="Churn Rate %", text=tb["Churn Rate %"].map("{:.1f}%".format),
        color="Churn Rate %", color_continuous_scale="RdYlGn_r",
        hover_data={"Count": True},
    )
    fig6.update_layout(height=300, coloraxis_showscale=False)
    fig6.update_traces(textposition="outside")
    st.plotly_chart(fig6, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 2 — DATA EXPLORER
# ─────────────────────────────────────────────
elif selected_page == "🔍 Data Explorer":
    st.title("🔍 Data Explorer")

    st.markdown("### Dataset Sample")
    st.dataframe(df_raw.head(50), use_container_width=True)

    st.markdown("### Data Shape & Validation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df_raw.shape[0]:,}")
    c2.metric("Columns", f"{df_raw.shape[1]}")
    c3.metric("Duplicate Rows", f"{df_raw.duplicated().sum()}")
    c4.metric("TotalCharges Fixed", f"{missing_tc}")

    st.markdown("### Missing Values After Cleaning")
    missing = df_raw.isnull().sum()
    missing_tc_blanks = (pd.to_numeric(df_raw["TotalCharges"], errors="coerce")).isna().sum()
    mv_df = pd.DataFrame({
        "Column": ["TotalCharges (blank strings)"] + list(missing[missing > 0].index),
        "Count": [missing_tc_blanks] + list(missing[missing > 0].values),
    })
    if mv_df.empty or (mv_df["Count"] == 0).all():
        st.success("✅ No missing values found (TotalCharges blanks imputed with MonthlyCharges).")
    else:
        st.dataframe(mv_df, use_container_width=True)
    st.info(f"ℹ️ {missing_tc_blanks} rows had blank TotalCharges — imputed using MonthlyCharges (all had tenure=0).")

    st.markdown("### Column Data Types & Unique Values")
    dtype_df = pd.DataFrame({
        "Column": df_raw.columns,
        "DType": df_raw.dtypes.values,
        "Unique Values": [df_raw[c].nunique() for c in df_raw.columns],
        "Sample Values": [str(df_raw[c].unique()[:4].tolist()) for c in df_raw.columns],
    })
    st.dataframe(dtype_df, use_container_width=True)

    st.markdown("### Descriptive Statistics (Numeric)")
    st.dataframe(df_raw.describe(), use_container_width=True)

    st.markdown("### Filter Data")
    churn_filter = st.selectbox("Filter by Churn", ["All", "Yes", "No"])
    contract_filter = st.multiselect("Filter by Contract", df["Contract"].unique().tolist(), default=df["Contract"].unique().tolist())
    filtered = df.copy()
    if churn_filter != "All":
        filtered = filtered[filtered["Churn"] == churn_filter]
    filtered = filtered[filtered["Contract"].isin(contract_filter)]
    st.dataframe(filtered[["customerID", "gender", "tenure", "MonthlyCharges",
                            "TotalCharges", "Contract", "InternetService",
                            "PaymentMethod", "Churn"]].reset_index(drop=True),
                 use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 3 — EDA
# ─────────────────────────────────────────────
elif selected_page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Demographics", "Services & Contract", "Charges & Tenure", "Correlation"]
    )

    with tab1:
        st.markdown("#### Churn by Demographic Variables")
        demo_cols = ["gender", "SeniorCitizen_Label", "Partner", "Dependents"]
        fig = make_subplots(rows=2, cols=2, subplot_titles=demo_cols)
        for i, col in enumerate(demo_cols):
            r, c = divmod(i, 2)
            grp = df.groupby([col, "Churn"]).size().reset_index(name="Count")
            for churn_val, color in CHURN_COLORS.items():
                subset = grp[grp["Churn"] == churn_val]
                fig.add_trace(
                    go.Bar(name=f"Churn={churn_val}", x=subset[col], y=subset["Count"],
                           marker_color=color, legendgroup=churn_val,
                           showlegend=(i == 0)),
                    row=r + 1, col=c + 1,
                )
        fig.update_layout(barmode="group", height=500, legend_title="Churn")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Churn Rate by Senior Citizen Status")
        sc = df.groupby("SeniorCitizen_Label")["Churn_Binary"].mean().reset_index()
        sc.columns = ["Senior Citizen", "Churn Rate"]
        sc["Churn Rate %"] = sc["Churn Rate"] * 100
        st.dataframe(sc, use_container_width=True)

    with tab2:
        st.markdown("#### Churn Rate by Service Features")
        service_cols = [
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod",
        ]
        rows_data = []
        for col in service_cols:
            grp = df.groupby(col)["Churn_Binary"].mean().reset_index()
            grp.columns = ["Value", "Churn Rate"]
            grp["Feature"] = col
            rows_data.append(grp)
        svc_df = pd.concat(rows_data)
        svc_df["Churn Rate %"] = svc_df["Churn Rate"] * 100

        selected_svc = st.selectbox("Select Feature", service_cols)
        svc_sub = svc_df[svc_df["Feature"] == selected_svc]
        fig2 = px.bar(
            svc_sub, x="Value", y="Churn Rate %",
            color="Churn Rate %", color_continuous_scale="RdYlGn_r",
            text=svc_sub["Churn Rate %"].map("{:.1f}%".format),
            title=f"Churn Rate by {selected_svc}",
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(coloraxis_showscale=False, height=380)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Churn Rate Heatmap — Internet × Contract")
        pivot = df.pivot_table(values="Churn_Binary", index="InternetService",
                                columns="Contract", aggfunc="mean")
        fig3 = px.imshow(
            pivot * 100, text_auto=".1f", color_continuous_scale="RdYlGn_r",
            labels={"color": "Churn Rate %"}, title="Churn Rate % (Internet × Contract)",
        )
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.markdown("#### Monthly Charges vs Tenure (coloured by Churn)")
        fig4 = px.scatter(
            df.sample(min(2000, len(df)), random_state=42),
            x="tenure", y="MonthlyCharges", color="Churn",
            color_discrete_map=CHURN_COLORS, opacity=0.5,
            labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly Charges ($)"},
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Total Charges Distribution by Churn")
            fig5 = px.histogram(
                df, x="TotalCharges", color="Churn", barmode="overlay",
                color_discrete_map=CHURN_COLORS, nbins=50, opacity=0.7,
            )
            fig5.update_layout(height=350)
            st.plotly_chart(fig5, use_container_width=True)
        with col2:
            st.markdown("#### Number of Services vs Churn Rate")
            ns = df.groupby("Num_Services")["Churn_Binary"].mean().reset_index()
            ns.columns = ["Num Services", "Churn Rate"]
            ns["Churn Rate %"] = ns["Churn Rate"] * 100
            fig6 = px.line(
                ns, x="Num Services", y="Churn Rate %", markers=True,
                title="Churn Rate by Number of Add-on Services",
            )
            fig6.update_layout(height=350)
            st.plotly_chart(fig6, use_container_width=True)

    with tab4:
        st.markdown("#### Correlation Matrix (Encoded Features)")
        _, _, feature_names, _, feat_df = prepare_model_data(df)
        feat_df["Churn_Binary"] = df["Churn_Binary"].values
        corr = feat_df.corr()
        fig7 = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, title="Feature Correlation Matrix",
        )
        fig7.update_layout(height=600)
        st.plotly_chart(fig7, use_container_width=True)

        st.markdown("#### Top Features Correlated with Churn")
        churn_corr = corr["Churn_Binary"].drop("Churn_Binary").sort_values(key=abs, ascending=False)
        fig8 = px.bar(
            x=churn_corr.values, y=churn_corr.index,
            orientation="h", color=churn_corr.values,
            color_continuous_scale="RdBu_r", range_color=[-1, 1],
            labels={"x": "Correlation", "y": "Feature"},
            title="Feature Correlation with Churn",
        )
        fig8.update_layout(height=500, coloraxis_showscale=False)
        st.plotly_chart(fig8, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 4 — BUSINESS INSIGHTS
# ─────────────────────────────────────────────
elif selected_page == "💡 Business Insights":
    st.title("💡 Business Insights — Answering Key Questions")
    st.info(
        "⚠️ Note: All findings below describe *associations* observed in the data. "
        "Correlation does not imply causation."
    )

    # Q1
    st.markdown("---")
    st.markdown("### Q1: What is the overall churn rate?")
    churn_rate = df["Churn_Binary"].mean() * 100
    retained = 100 - churn_rate
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
    col2.metric("Churned Customers", f"{df['Churn_Binary'].sum():,}")
    col3.metric("Retained Customers", f"{(df['Churn_Binary'] == 0).sum():,}")
    st.markdown(
        f"**Finding:** {churn_rate:.1f}% of the {len(df):,} customers in this dataset churned. "
        "This rate (~26%) is notably high for a telecom provider and represents substantial "
        "monthly revenue at risk."
    )

    # Q2
    st.markdown("---")
    st.markdown("### Q2: Which customer segments have the highest churn?")
    seg_cols = ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen_Label",
                "Tenure_Band", "Charge_Band"]
    seg_data = []
    for col in seg_cols:
        g = df.groupby(col, observed=True)["Churn_Binary"].agg(["mean", "count"]).reset_index()
        g.columns = ["Segment Value", "Churn Rate", "Count"]
        g["Category"] = col
        g["Churn Rate %"] = g["Churn Rate"] * 100
        seg_data.append(g)
    seg_df = pd.concat(seg_data)
    fig_q2 = px.bar(
        seg_df, x="Segment Value", y="Churn Rate %", color="Category",
        facet_col="Category", facet_col_wrap=3,
        color_discrete_sequence=PALETTE,
        text=seg_df["Churn Rate %"].map("{:.0f}%".format),
    )
    fig_q2.update_traces(textposition="outside")
    fig_q2.update_layout(height=600, showlegend=False)
    fig_q2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig_q2, use_container_width=True)
    st.markdown(
        "**Finding:** Month-to-month contract customers, Fiber optic internet users, "
        "customers paying by electronic check, senior citizens, and short-tenure customers "
        "show the highest observed churn rates in this dataset."
    )

    # Q3
    st.markdown("---")
    st.markdown("### Q3: How do tenure and monthly charges relate to churn?")
    col1, col2 = st.columns(2)
    with col1:
        avg_stats = df.groupby("Churn")[["tenure", "MonthlyCharges", "TotalCharges"]].mean().T
        avg_stats.columns = ["Retained", "Churned"]
        avg_stats["Difference"] = avg_stats["Churned"] - avg_stats["Retained"]
        st.dataframe(avg_stats.style.format("{:.2f}"), use_container_width=True)
    with col2:
        fig_q3 = px.violin(
            df, x="Churn", y="tenure", color="Churn",
            color_discrete_map=CHURN_COLORS, box=True, points=False,
            labels={"tenure": "Tenure (months)"},
            title="Tenure Distribution by Churn",
        )
        fig_q3.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_q3, use_container_width=True)
    st.markdown(
        "**Finding:** Churned customers had substantially shorter average tenures (~18 mo vs ~38 mo for retained) "
        "and higher average monthly charges (~$74 vs ~$61). This association suggests that newer customers "
        "paying more per month are more likely to be observed in the churned group."
    )

    # Q4
    st.markdown("---")
    st.markdown("### Q4: How does contract type affect churn patterns?")
    ct_detail = df.groupby(["Contract", "Tenure_Band"], observed=True)["Churn_Binary"].mean().reset_index()
    ct_detail.columns = ["Contract", "Tenure Band", "Churn Rate"]
    ct_detail["Churn Rate %"] = ct_detail["Churn Rate"] * 100
    fig_q4 = px.bar(
        ct_detail, x="Tenure Band", y="Churn Rate %", color="Contract",
        barmode="group", color_discrete_sequence=PALETTE,
        title="Churn Rate by Contract Type and Tenure Band",
    )
    fig_q4.update_layout(height=380)
    st.plotly_chart(fig_q4, use_container_width=True)
    ct_summary = df.groupby("Contract")["Churn_Binary"].agg(["mean", "count"]).reset_index()
    ct_summary.columns = ["Contract", "Churn Rate", "Customers"]
    ct_summary["Churn Rate %"] = ct_summary["Churn Rate"] * 100
    st.dataframe(ct_summary[["Contract", "Churn Rate %", "Customers"]], use_container_width=True)
    st.markdown(
        "**Finding:** Month-to-month customers churn at a much higher rate (~43%) compared to "
        "one-year (~11%) and two-year (~3%) contract holders. This pattern holds across tenure bands."
    )

    # Q5
    st.markdown("---")
    st.markdown("### Q5: How do internet service and payment methods relate to churn?")
    col1, col2 = st.columns(2)
    with col1:
        isp = df.groupby("InternetService")["Churn_Binary"].mean().reset_index()
        isp["Churn Rate %"] = isp["Churn_Binary"] * 100
        fig5a = px.bar(isp, x="InternetService", y="Churn Rate %",
                       color="Churn Rate %", color_continuous_scale="RdYlGn_r",
                       text=isp["Churn Rate %"].map("{:.1f}%".format),
                       title="Internet Service vs Churn")
        fig5a.update_traces(textposition="outside")
        fig5a.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig5a, use_container_width=True)
    with col2:
        pm = df.groupby("PaymentMethod")["Churn_Binary"].mean().reset_index()
        pm["Churn Rate %"] = pm["Churn_Binary"] * 100
        fig5b = px.bar(pm, x="PaymentMethod", y="Churn Rate %",
                       color="Churn Rate %", color_continuous_scale="RdYlGn_r",
                       text=pm["Churn Rate %"].map("{:.1f}%".format),
                       title="Payment Method vs Churn")
        fig5b.update_traces(textposition="outside")
        fig5b.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig5b, use_container_width=True)
    st.markdown(
        "**Finding:** Fiber optic users show a churn rate of ~41%, compared to ~19% for DSL and ~7% for no internet. "
        "Customers using electronic check payment show the highest churn rate (~45%), while "
        "credit card (automatic) and bank transfer (automatic) customers show lower rates (~15–16%)."
    )

    # Q6
    st.markdown("---")
    st.markdown("### Q6: Which services are associated with different churn rates?")
    add_on_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]
    svc_rates = []
    for col in add_on_cols:
        for val in ["Yes", "No"]:
            sub = df[df[col] == val]
            rate = sub["Churn_Binary"].mean() * 100
            svc_rates.append({"Service": col, "Subscribed": val, "Churn Rate %": rate, "Count": len(sub)})
    svc_rates_df = pd.DataFrame(svc_rates)
    fig_q6 = px.bar(
        svc_rates_df, x="Service", y="Churn Rate %", color="Subscribed",
        barmode="group", color_discrete_map={"Yes": "#4CAF50", "No": "#F44336"},
        title="Churn Rate: Subscribed vs Not Subscribed to Add-on Services",
        text=svc_rates_df["Churn Rate %"].map("{:.1f}%".format),
    )
    fig_q6.update_traces(textposition="outside")
    fig_q6.update_layout(height=400)
    st.plotly_chart(fig_q6, use_container_width=True)
    st.markdown(
        "**Finding:** Customers *without* OnlineSecurity, TechSupport, OnlineBackup, and DeviceProtection "
        "show notably higher churn rates than those who subscribe to these services. "
        "StreamingTV and StreamingMovies show less differentiation in churn rates."
    )

    # Q7 — preview (full answer on model page)
    st.markdown("---")
    st.markdown("### Q7: Which features are most important to the prediction model?")
    st.info("➡️ Feature importance analysis is available on the **Model Training & Evaluation** page after models are trained.")

# ─────────────────────────────────────────────
# PAGE 5 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
elif selected_page == "⚙️ Feature Engineering":
    st.title("⚙️ Feature Engineering")

    st.markdown(
        "The following additional features were engineered from the raw dataset "
        "to improve model interpretability and predictive power."
    )

    st.markdown("### Engineered Features")
    feat_table = pd.DataFrame({
        "Feature": ["Tenure_Band", "Charge_Band", "Num_Services", "Charge_Per_Month"],
        "Description": [
            "Binned tenure into 4 bands: 0–12, 13–24, 25–48, 49–72 months",
            "Binned MonthlyCharges: Low (<35), Mid (35–65), High (65–95), Very High (>95)",
            "Count of add-on services subscribed (OnlineSecurity, Backup, DeviceProtection, TechSupport, StreamingTV, Movies)",
            "TotalCharges / (tenure + 1) — average monthly spend controlling for tenure",
        ],
        "Type": ["Categorical (ordinal)", "Categorical (ordinal)", "Numeric (integer)", "Numeric (float)"],
    })
    st.dataframe(feat_table, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Churn Rate by Num_Services")
        ns = df.groupby("Num_Services")["Churn_Binary"].agg(["mean", "count"]).reset_index()
        ns.columns = ["Num Services", "Churn Rate", "Count"]
        ns["Churn Rate %"] = ns["Churn Rate"] * 100
        fig1 = px.bar(ns, x="Num Services", y="Churn Rate %",
                      text=ns["Churn Rate %"].map("{:.1f}%".format),
                      color="Churn Rate %", color_continuous_scale="RdYlGn_r")
        fig1.update_traces(textposition="outside")
        fig1.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.markdown("#### Charge_Per_Month Distribution by Churn")
        fig2 = px.box(df, x="Churn", y="Charge_Per_Month", color="Churn",
                      color_discrete_map=CHURN_COLORS,
                      labels={"Charge_Per_Month": "Avg Charge/Month ($)"})
        fig2.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Churn Rate by Tenure Band and Charge Band")
    pivot2 = df.pivot_table(
        values="Churn_Binary", index="Tenure_Band", columns="Charge_Band",
        aggfunc="mean", observed=True
    )
    fig3 = px.imshow(
        pivot2 * 100, text_auto=".1f", color_continuous_scale="RdYlGn_r",
        labels={"color": "Churn Rate %"},
        title="Churn Rate % (Tenure Band × Charge Band)",
    )
    fig3.update_layout(height=380)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Model Feature Set")
    st.markdown(
        "Features used for model training (engineered band/count features are *excluded* from the model "
        "to avoid data leakage — only raw numeric + encoded categorical features are used):"
    )
    _, _, feature_names, _, _ = prepare_model_data(df)
    st.code(", ".join(feature_names))

# ─────────────────────────────────────────────
# PAGE 6 — MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
elif selected_page == "🤖 Model Training & Evaluation":
    st.title("🤖 Model Training & Evaluation")

    with st.spinner("Training models... (cached after first run)"):
        results, trained, X_train, X_test, y_train, y_test, scaler, feature_names, le_dict = train_models(df)

    st.success(f"✅ {len(results)} models trained on {len(X_train)} samples | Tested on {len(X_test)} samples")

    # Metrics table
    st.markdown("### Model Performance Summary")
    metrics_df = pd.DataFrame(
        {m: {k: v for k, v in r.items() if k not in ("y_pred", "y_prob", "cm")}
         for m, r in results.items()}
    ).T.reset_index()
    metrics_df.columns = ["Model"] + [c for c in metrics_df.columns if c != "index" and c != "Model"]
    metrics_df = metrics_df.rename(columns={"index": "Model"})
    for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        metrics_df[col] = metrics_df[col].astype(float)

    st.dataframe(
        metrics_df.style
            .format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]})
            .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                           color="#c6efce"),
        use_container_width=True,
    )

    # Select model to inspect
    selected_model = st.selectbox("Select model to inspect", list(results.keys()))
    res = results[selected_model]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### Confusion Matrix — {selected_model}")
        cm = res["cm"]
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            x=["No Churn", "Churn"], y=["No Churn", "Churn"],
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.markdown(f"#### ROC Curve — {selected_model}")
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={res['ROC-AUC']:.4f}",
                                     line=dict(color="#2196F3", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(color="gray", dash="dash"), name="Random"))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            height=380, legend=dict(x=0.6, y=0.1),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("#### Classification Report")
    cr = classification_report(y_test, res["y_pred"], target_names=["No Churn", "Churn"], output_dict=True)
    cr_df = pd.DataFrame(cr).T
    st.dataframe(cr_df.style.format("{:.4f}", na_rep="-"), use_container_width=True)

    # Feature importance
    st.markdown("---")
    st.markdown("### 🔎 Feature Importance / Model Interpretation")
    st.caption("(Q7: Which features are most important to the prediction model?)")

    model_obj, uses_scaled = trained[selected_model]

    importances = None
    imp_type = ""

    if hasattr(model_obj, "feature_importances_"):
        importances = model_obj.feature_importances_
        imp_type = "Tree-based feature importance (mean decrease in impurity)"
    elif hasattr(model_obj, "coef_"):
        importances = np.abs(model_obj.coef_[0])
        imp_type = "Logistic Regression |coefficient| magnitude (scaled features)"
    else:
        # Permutation importance for KNN
        Xte = scaler.transform(X_test) if uses_scaled else X_test
        pi = permutation_importance(model_obj, Xte, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importances = pi.importances_mean
        imp_type = "Permutation importance (mean accuracy drop)"

    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=False).head(20)

    fig_imp = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
        title=f"Top 20 Features — {selected_model}<br><sup>{imp_type}</sup>",
    )
    fig_imp.update_layout(height=550, yaxis=dict(autorange="reversed"),
                          coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown(
        "**Finding (Q7):** Across tree-based models, `tenure`, `TotalCharges`, `MonthlyCharges`, and "
        "`Contract` consistently emerge as the most important features for predicting churn. "
        "`InternetService` type and `PaymentMethod` also rank highly. "
        "Demographic features like `gender` and `Dependents` show relatively low importance."
    )

# ─────────────────────────────────────────────
# PAGE 7 — MODEL COMPARISON
# ─────────────────────────────────────────────
elif selected_page == "📈 Model Comparison":
    st.title("📈 Model Comparison")

    with st.spinner("Loading model results..."):
        results, trained, X_train, X_test, y_train, y_test, scaler, feature_names, le_dict = train_models(df)

    metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    comp_data = {m: {k: results[m][k] for k in metrics_list} for m in results}
    comp_df = pd.DataFrame(comp_data).T.reset_index().rename(columns={"index": "Model"})

    # Grouped bar
    st.markdown("### Metric Comparison Across Models")
    melted = comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig1 = px.bar(
        melted, x="Metric", y="Score", color="Model",
        barmode="group", color_discrete_sequence=PALETTE,
        text=melted["Score"].map("{:.3f}".format),
    )
    fig1.update_traces(textposition="outside")
    fig1.update_layout(height=450, yaxis_range=[0, 1.1])
    st.plotly_chart(fig1, use_container_width=True)

    # Radar chart
    st.markdown("### Radar Chart — Model Performance Profile")
    fig2 = go.Figure()
    for i, (model_name, row) in enumerate(comp_df.iterrows()):
        vals = [row[m] for m in metrics_list]
        vals += [vals[0]]
        fig2.add_trace(go.Scatterpolar(
            r=vals, theta=metrics_list + [metrics_list[0]],
            fill="toself", name=row["Model"],
            line_color=PALETTE[i % len(PALETTE)],
        ))
    fig2.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=500)
    st.plotly_chart(fig2, use_container_width=True)

    # All ROC curves
    st.markdown("### ROC Curves — All Models")
    fig3 = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        fig3.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={res['ROC-AUC']:.4f})",
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
        ))
    fig3.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              line=dict(color="gray", dash="dash"), name="Random"))
    fig3.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=500,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Summary table
    st.markdown("### Summary Table")
    for col in metrics_list:
        comp_df[col] = comp_df[col].astype(float)
    best_model = comp_df.loc[comp_df["ROC-AUC"].idxmax(), "Model"]
    st.dataframe(
        comp_df.set_index("Model").style
            .format("{:.4f}")
            .highlight_max(axis=0, color="#c6efce"),
        use_container_width=True,
    )
    st.success(f"🏆 Best model by ROC-AUC: **{best_model}**")

# ─────────────────────────────────────────────
# PAGE 8 — CUSTOMER PREDICTION INTERFACE
# ─────────────────────────────────────────────
elif selected_page == "🔮 Customer Prediction":
    st.title("🔮 Interactive Customer Churn Prediction")
    st.markdown(
        "Enter a customer's details below to predict their likelihood of churning. "
        "The prediction uses the same preprocessing pipeline as model training."
    )

    with st.spinner("Loading trained models..."):
        results, trained, X_train, X_test, y_train, y_test, scaler, feature_names, le_dict = train_models(df)

    # Determine best model (by ROC-AUC on test set)
    best_name = max(results, key=lambda k: results[k]["ROC-AUC"])

    st.sidebar.markdown("---")
    pred_model_name = st.sidebar.selectbox(
        "Prediction Model", list(trained.keys()),
        index=list(trained.keys()).index(best_name),
    )

    st.markdown(f"**Active prediction model:** `{pred_model_name}` | Test ROC-AUC: `{results[pred_model_name]['ROC-AUC']:.4f}`")
    st.markdown("---")

    with st.form("prediction_form"):
        st.markdown("### Customer Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with c2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        with c3:
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        c4, c5, c6 = st.columns(3)
        with c4:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        with c5:
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
        with c6:
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                             float(round(monthly_charges * max(tenure, 1), 2)),
                                             step=1.0)

        submitted = st.form_submit_button("🔮 Predict Churn", type="primary")

    if submitted:
        # Build input row matching training feature order
        senior_int = 1 if senior == "Yes" else 0

        input_dict = {
            "gender": gender,
            "SeniorCitizen": senior_int,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        # Encode using the same LabelEncoders fitted on training data
        input_df = pd.DataFrame([input_dict])
        for col, le in le_dict.items():
            if col in input_df.columns:
                val = input_df[col].iloc[0]
                if val in le.classes_:
                    input_df[col] = le.transform([val])
                else:
                    input_df[col] = le.transform([le.classes_[0]])

        # Ensure column order matches training
        input_arr = input_df[feature_names].values

        model_obj, uses_scaled = trained[pred_model_name]
        if uses_scaled:
            input_arr = scaler.transform(input_arr)

        prediction = model_obj.predict(input_arr)[0]
        probability = model_obj.predict_proba(input_arr)[0]

        churn_prob = probability[1]
        no_churn_prob = probability[0]

        st.markdown("---")
        st.markdown("### 📊 Prediction Result")

        res_col1, res_col2, res_col3 = st.columns(3)
        pred_label = "🔴 WILL CHURN" if prediction == 1 else "🟢 WILL NOT CHURN"
        res_col1.metric("Prediction", pred_label)
        res_col2.metric("Churn Probability", f"{churn_prob:.1%}")
        res_col3.metric("Retention Probability", f"{no_churn_prob:.1%}")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob * 100,
            number={"suffix": "%", "font": {"size": 32}},
            title={"text": "Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#F44336" if churn_prob >= 0.5 else "#4CAF50"},
                "steps": [
                    {"range": [0, 30], "color": "#E8F5E9"},
                    {"range": [30, 60], "color": "#FFF9C4"},
                    {"range": [60, 100], "color": "#FFEBEE"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Risk tier
        if churn_prob < 0.30:
            risk_tier = "🟢 Low Risk"
            rec = "Customer shows low churn risk. Standard retention practices apply."
        elif churn_prob < 0.60:
            risk_tier = "🟡 Medium Risk"
            rec = "Consider proactive outreach, loyalty offers, or contract upgrade incentives."
        else:
            risk_tier = "🔴 High Risk"
            rec = "High churn risk. Prioritise immediate retention intervention — targeted discount, contract upgrade, or personalised support."

        st.markdown(f"**Risk Tier:** {risk_tier}")
        st.info(f"💡 **Business Recommendation:** {rec}")

        # Show input summary
        with st.expander("View input data sent to model"):
            st.json(input_dict)
