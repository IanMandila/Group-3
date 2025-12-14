import streamlit as st
import plotly.express as px
from utils.data_loader import load_raw_data
from utils.preprocessing import clean_data

st.set_page_config(
    page_title="Adult Income Enterprise BI Dashboard",
    layout="wide"
)

DATA_PATH = "data/BI Dataset.xlsx"
df_raw = load_raw_data(DATA_PATH)
df = clean_data(df_raw)

# detect gender/sex column
gender_col = None
for _c in ("gender", "sex"):
    if _c in df.columns:
        gender_col = _c
        break

st.sidebar.title("📊 Analysis Dashboard")
section = st.sidebar.radio(
    "Navigate",
    [
        "income",
        "income vs occupation",
        "Education vs Income",
        "Gender vs Income",
        "Hours vs Income",
        "Conclusion",
    ],
)

st.markdown("""
<div style="text-align:center; padding:15px">
<h1>🏦 Adult Income Enterprise Intelligence Dashboard</h1>
<p>BIT 2119 – Business Intelligence & Decision Support Systems</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{len(df):,}")
c2.metric("Average Age", round(df["age"].mean(), 1))
c3.metric("Avg Weekly Hours", round(df["hours_per_week"].mean(), 1))
c4.metric("High Income Rate", f"{df['income_binary'].mean() * 100:.1f}%")

if section == "income":
    st.markdown("""
    <div style="background-color:#064e3b; padding:15px; border-radius:8px">
    <b>Business Context</b><br><br>
    Financial institutions require accurate income classification
    to support credit scoring and loan pre-qualification.
    </div>
    """, unsafe_allow_html=True)

    fig_income = px.pie(df, names="income", hole=0.5, title="Income Distribution")
    st.plotly_chart(fig_income, use_container_width=True)

elif section == "income vs occupation":
    fig_job = px.histogram(df, x="occupation", color="income", title="Income by Occupation")
    fig_job.update_xaxes(tickangle=45)
    st.plotly_chart(fig_job, use_container_width=True)

elif section == "Education vs Income":
    fig_edu = px.histogram(df, x="education", color="income", title="Education vs Income")
    fig_edu.update_xaxes(tickangle=45)
    st.plotly_chart(fig_edu, use_container_width=True)


elif section == "Gender vs Income":
    if gender_col is not None:
        counts = df.groupby([gender_col, "income"]).size().reset_index(name="count")
        fig_gender_bar = px.bar(
            counts,
            x=gender_col,
            y="count",
            color="income",
            barmode="group",
            title="Income by Gender (Counts)",
            labels={"count": "Number of Records"},
        )
        st.plotly_chart(fig_gender_bar, use_container_width=True)
    else:
        st.info("No `gender`/`sex` column found in the dataset; cannot show gender breakdown.")

elif section == "Hours vs Income":
    hours_mean = df.groupby("income")["hours_per_week"].mean().reset_index()
    fig_hours_bar = px.bar(
        hours_mean,
        x="income",
        y="hours_per_week",
        title="Average Weekly Hours by Income",
        labels={"hours_per_week": "Avg Hours per Week", "income": "Income"},
    )
    st.plotly_chart(fig_hours_bar, use_container_width=True)

elif section == "Conclusion":
    st.success(
        "Education, age, and work intensity strongly influence income.\n"
        "BI-driven income classification improves fair credit decisions."
    )

    st.markdown(
        """
        In financial product design, income trends associated with age and education suggest that younger individuals and those with lower educational attainment are more suitable for entry-level savings products and low-risk credit facilities. Middle-aged and better-educated individuals, who exhibit higher income stability, align well with salary-backed loans, asset financing, and long-term investment products. Kenyan banks and SACCOs can therefore segment customers by life-cycle stage to improve product relevance.
        """
    )

    st.markdown(
        """
        For microcredit and lending risk assessment, higher education levels, stable employment sectors, and consistent working hours correlate with reduced income volatility. In Kenya, where informal employment is widespread, these indicators can support alternative credit scoring models, enabling microfinance institutions and mobile lenders to manage risk while promoting financial inclusion.
        """
    )

    st.markdown(
        """
        In labour-market profiling, disparities in income across education levels and occupations highlight the need for targeted skills development. Policymakers can use such insights to identify vulnerable workforce segments and design vocational training programmes.
        """
    )

    st.markdown(
        """
        From a regulatory perspective, income patterns linked to demographic and employment variables support fair lending oversight, wage equity monitoring, and evidence-based socio-economic reporting aligned with Kenya’s development objectives.
        """
    )

with st.expander("View Cleaned Dataset"):
    st.dataframe(df.head(100))
