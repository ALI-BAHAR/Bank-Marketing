"""Interactive version of the analysis. Run with:  streamlit run app.py"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

import bank_analysis as ba

st.set_page_config(page_title="Bank Marketing Analysis", layout="wide")
BLUE, GREY = "#2F5D8A", "#9AA5B1"


@st.cache_data
def load():
    raw = ba.load_raw()
    df, log = ba.clean(raw)
    return raw, df, log


@st.cache_data
def run_models(df):
    X, y = ba.features_target(df)
    results, probas = ba.compare_models(X, y)
    return y, results, probas


raw, df, log = load()
g = ba.add_groups(df)
avg = 100 * df["subscribed"].mean()

st.title("Bank Marketing: who subscribes to a term deposit?")
st.write("Phone campaigns of a Portuguese bank ([UCI dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing), "
         f"10% sample). **{len(df):,} clients** after cleaning, **{avg:.1f}%** subscribed.")

tab_data, tab_explore, tab_model = st.tabs(["Data and cleaning", "Explore", "Model"])

with tab_data:
    st.subheader("Missing values ('unknown') in the original data")
    st.dataframe(ba.missing_report(raw))
    st.subheader("Cleaning steps")
    for step in log:
        st.markdown(f"- {step}")
    with st.expander("Show the cleaned data"):
        st.dataframe(df)

with tab_explore:
    options = {"Month of last contact": ("month", ba.MONTH_ORDER),
               "Number of calls in this campaign": ("contacts_in_campaign", None),
               "Outcome of the previous campaign": ("poutcome", None),
               "Age group": ("age_group", None),
               "Job": ("job", None),
               "Contact type": ("contact", None),
               "Education": ("education", None),
               "Marital status": ("marital", None)}
    choice = st.selectbox("Subscription rate by", list(options))
    col, order = options[choice]
    t = ba.rate_by(g, col, order)
    if order is None and col in ("job", "education"):
        t = t.sort_values("rate", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar([str(i) for i in t.index], t["rate"], color=BLUE)
    ax.axhline(avg, ls="--", color=GREY, label=f"Average {avg:.1f}%")
    for i, (r, n) in enumerate(zip(t["rate"], t["clients"])):
        ax.text(i, r + 0.5, f"{r:.0f}%\nn={n}", ha="center", fontsize=8)
    ax.set_ylabel("Subscription rate (%)")
    ax.set_ylim(0, t["rate"].max() * 1.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    plt.xticks(rotation=30 if len(t) > 6 else 0, ha="right" if len(t) > 6 else "center")
    st.pyplot(fig)
    st.caption("n = number of clients in the group. Rates of small groups are less reliable.")

with tab_model:
    st.write("Models are evaluated with 5-fold stratified cross-validation. "
             "`duration` is excluded because it is only known after the call.")
    with st.spinner("Training models..."):
        y, results, probas = run_models(df)
    st.subheader("Model comparison")
    st.dataframe(results.style.format("{:.3f}"))

    name = st.radio("Model", ["Logistic regression (balanced)", "KNN (k=6)"], horizontal=True)
    threshold = st.slider("Decision threshold (probability above which a client is flagged as 'yes')",
                          0.05, 0.95, 0.5, 0.05)
    s = ba.scores(y, probas[name], threshold)
    c1, c2, c3 = st.columns(3)
    c1.metric("Recall (subscribers found)", f"{100 * s['Recall (yes)']:.0f}%")
    c2.metric("Precision (flagged who subscribe)", f"{100 * s['Precision (yes)']:.0f}%")
    c3.metric("Accuracy", f"{100 * s['Accuracy']:.0f}%")
    cm = pd.DataFrame(confusion_matrix(y, (probas[name] >= threshold).astype(int)),
                      index=["Actual no", "Actual yes"], columns=["Predicted no", "Predicted yes"])
    st.dataframe(cm)

    st.subheader("If the bank calls only the top-scored clients")
    st.dataframe(ba.gains(y, probas[name]), hide_index=True)
