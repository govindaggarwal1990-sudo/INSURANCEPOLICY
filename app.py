
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px

from model_utils import (
    prepare_data, evaluate_models, run_inference_on_new_data
)

st.set_page_config(page_title="Insurance Workforce Attrition — Analytics", layout="wide")

# Sidebar upload
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV (must include Attrition/attribution)", type=["csv"])

@st.cache_data(show_spinner=False)
def load_df(file):
    return pd.read_csv(file)

if uploaded is not None:
    df = load_df(uploaded)
else:
    st.info("Upload your dataset to begin. The app will look for a target column 'Attrition' (case-insensitive).")
    st.stop()

# Locate target
possible_targets = ["Attrition", "attrition", "Attribution", "attribution"]
target_col = None
for c in df.columns:
    if c in possible_targets or c.lower() in {"attrition", "attribution"}:
        target_col = c; break
if target_col is None:
    st.error("Target column 'Attrition' (or 'attribution') not found."); st.stop()

Xcols = [c for c in df.columns if c != target_col]
cat_cols = df[Xcols].select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df[Xcols].select_dtypes(include=[np.number, "bool"]).columns.tolist()

# Filters
st.sidebar.header("Filters (apply to all charts)")
job_role_candidates = [c for c in ["JobRole","Role","Designation","JobTitle","Function","Department"] if c in df.columns]
job_col = job_role_candidates[0] if len(job_role_candidates)>0 else (cat_cols[0] if len(cat_cols)>0 else None)
if job_col:
    options = sorted(df[job_col].dropna().astype(str).unique().tolist())
    selected_jobs = st.sidebar.multiselect(f"Filter by {job_col}", options, default=options[:min(8,len(options))])
else:
    selected_jobs = []

sat_candidates = [c for c in df.columns if "satisfaction" in c.lower() or c in ["JobSatisfaction","EnvironmentSatisfaction","RelationshipSatisfaction","WorkLifeBalance","CustomerSatisfaction"]]
sat_col = sat_candidates[0] if len(sat_candidates)>0 else None
threshold = None
if sat_col is not None and pd.api.types.is_numeric_dtype(df[sat_col]):
    vmin, vmax = float(pd.to_numeric(df[sat_col], errors="coerce").min()), float(pd.to_numeric(df[sat_col], errors="coerce").max())
    if np.isfinite(vmin) and np.isfinite(vmax):
        threshold = st.sidebar.slider(f"Min {sat_col}", float(vmin), float(vmax), float(vmin))

filtered = df.copy()
if job_col and len(selected_jobs)>0:
    filtered = filtered[filtered[job_col].astype(str).isin([str(x) for x in selected_jobs])]
if sat_col is not None and threshold is not None:
    filtered = filtered[pd.to_numeric(filtered[sat_col], errors="coerce") >= threshold]

st.title("Insurance Workforce Attrition — Decision Dashboard")
st.caption("For insurance providers to monitor, model, and mitigate employee attrition.")

tab1, tab2, tab3 = st.tabs(["📊 Insights", "🤖 Modeling (DT / RF / GB)", "📤 Predict on New Data"])

def pick_cat(exclude=None):
    exclude = set(exclude or []); avail = [c for c in cat_cols if c not in exclude]
    return avail[0] if len(avail)>0 else None

def pick_num(exclude=None):
    exclude = set(exclude or []); avail = [c for c in num_cols if c not in exclude]
    return avail[0] if len(avail)>0 else None

with tab1:
    st.subheader("Overview & Filters")
    c1,c2,c3 = st.columns(3)
    c1.metric("Employees (after filters)", len(filtered))
    if filtered[target_col].nunique()==2:
        pos_rate = (filtered[target_col] == filtered[target_col].unique()[-1]).mean()
        c2.metric("Attrition Rate", f"{100*pos_rate:.1f}%")
    else:
        c2.metric("Attrition Rate", "—")
    c3.metric("Columns", len(filtered.columns))

    # 1) Job Role vs Attrition
    st.markdown("#### 1) Attrition rate by Job Role")
    if job_col:
        df1 = filtered.groupby([job_col, target_col]).size().reset_index(name="count")
        df1["total"] = df1.groupby(job_col)["count"].transform("sum")
        df1["rate"] = df1["count"] / df1["total"]
        chart1 = alt.Chart(df1).mark_bar().encode(
            x=alt.X(f"{job_col}:N", title=job_col),
            y=alt.Y("rate:Q", stack="normalize", axis=alt.Axis(format='%')),
            color=alt.Color(f"{target_col}:N"),
            tooltip=[job_col, target_col, alt.Tooltip("rate:Q", format=".1%"), "count:Q"]
        ).properties(height=360)
        st.altair_chart(chart1, use_container_width=True)
    else:
        st.info("No job role column to plot.")

    # 2) Income/numeric distribution by Attrition
    st.markdown("#### 2) Pay / Numeric distribution by Attrition")
    income_col = None
    for c in ["MonthlyIncome","Salary","Wage","Pay","Premium"]:
        if c in filtered.columns and pd.api.types.is_numeric_dtype(filtered[c]):
            income_col = c; break
    income_col = income_col or pick_num()
    if income_col:
        st.plotly_chart(px.box(filtered, x=target_col, y=income_col, points="all", title=f"{income_col} by Attrition"), use_container_width=True)
    else:
        st.info("No numeric column found for box plot.")

    # 3) Heatmap: two categoricals
    st.markdown("#### 3) Heatmap: Attrition rate by two categorical features")
    catA = None
    for cand in ["OverTime","Shift","Region","Branch","Team","Location"]:
        if cand in filtered.columns: catA = cand; break
    if catA is None: catA = pick_cat(exclude=[job_col])
    catB = None
    for cand in ["JobLevel","Grade","Seniority","ExperienceBand","Channel"]:
        if cand in filtered.columns: catB = cand; break
    if catB is None: catB = pick_cat(exclude=[job_col, catA])
    if catA and catB and filtered[target_col].nunique()==2:
        pos = filtered[target_col].unique()[-1]
        rate = (filtered.assign(pos=(filtered[target_col]==pos).astype(int))
                        .groupby([catA,catB])["pos"].mean().reset_index(name="rate"))
        heatfig = px.density_heatmap(rate, x=catA, y=catB, z="rate", color_continuous_scale="Blues")
        heatfig.update_layout(coloraxis_colorbar_title="Attrition rate")
        st.plotly_chart(heatfig, use_container_width=True)
    else:
        st.info("Need two categorical columns for heatmap.")

    # 4) Commute/Distance vs Attrition (violin)
    st.markdown("#### 4) Commute / Distance vs Attrition (violin)")
    dist_col = None
    for c in ["DistanceFromHome","CommuteTime","TravelTime","TravelDistance"]:
        if c in filtered.columns and pd.api.types.is_numeric_dtype(filtered[c]):
            dist_col = c; break
    dist_col = dist_col or pick_num(exclude=[income_col])
    if dist_col:
        st.plotly_chart(px.violin(filtered, x=target_col, y=dist_col, box=True, points="all", title=f"{dist_col} by Attrition"), use_container_width=True)
    else:
        st.info("No numeric column suitable for violin.")

    # 5) Tenure/Age bins vs Attrition
    st.markdown("#### 5) Attrition rate by tenure/age bins")
    bin_col = None
    for c in ["YearsAtCompany","Tenure","YearsOfService","Age"]:
        if c in filtered.columns and pd.api.types.is_numeric_dtype(filtered[c]):
            bin_col = c; break
    bin_col = bin_col or pick_num(exclude=[income_col, dist_col])
    if bin_col and filtered[target_col].nunique()==2:
        q = pd.qcut(filtered[bin_col].rank(method="first"), q=5, labels=["Q1","Q2","Q3","Q4","Q5"])
        tmp = filtered.copy(); tmp["Bin"] = q
        pos = filtered[target_col].unique()[-1]
        rate_df = tmp.assign(pos=(tmp[target_col]==pos).astype(int)).groupby("Bin")["pos"].mean().reset_index(name="AttritionRate")
        line = alt.Chart(rate_df).mark_line(point=True).encode(
            x=alt.X("Bin:N", title=f"{bin_col} (quintiles)"),
            y=alt.Y("AttritionRate:Q", axis=alt.Axis(format='%')),
            tooltip=["Bin", alt.Tooltip("AttritionRate:Q", format=".1%")]
        ).properties(height=360)
        st.altair_chart(line, use_container_width=True)
    else:
        st.info("No suitable numeric column for binning plot.")

with tab2:
    st.subheader("Train & Evaluate — Decision Tree, Random Forest, Gradient Boosting (5-fold CV)")
    c1, c2 = st.columns([1,3])
    with c1:
        test_size = st.slider("Test size", 0.1, 0.4, value=0.2, step=0.05)
        random_state = st.number_input("Random state", value=42, step=1)
        if st.button("Run 5-fold CV + Train Models"):
            with st.spinner("Training models with Stratified 5-fold CV..."):
                X, y, classes = prepare_data(df, target_col)
                results, plots, fitted = evaluate_models(X, y, classes, test_size=test_size, random_state=random_state, cv_splits=5)
                st.session_state["results"] = results
                st.session_state["plots"] = plots
                st.session_state["fitted"] = fitted
    if "results" in st.session_state:
        st.success("Models trained. See the outputs below.")
        st.dataframe(st.session_state["results"])
        for label, fig in st.session_state["plots"].items():
            st.markdown(f"**{label}**"); st.pyplot(fig)

with tab3:
    st.subheader("Predict on New Data")
    st.caption("Upload a new CSV and select one of the trained models to get predictions & probabilities.")
    newfile = st.file_uploader("Upload new CSV for prediction", key="predict_uploader", type=["csv"])
    if newfile is not None:
        if "fitted" not in st.session_state:
            st.warning("Please train models in the Modeling tab first.")
        else:
            newdf = pd.read_csv(newfile)
            model_name = st.selectbox("Choose a trained model", options=list(st.session_state["fitted"].keys()))
            yhat, proba, out = run_inference_on_new_data(newdf, st.session_state["fitted"][model_name], proba_colname="Attrition_Prob")
            st.dataframe(out.head(50))
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv, file_name="predictions_with_attrition.csv", mime="text/csv")
