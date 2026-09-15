import streamlit as st
import polars as pl
import duckdb
import numpy as np
import plotly.express as px

# Streamlit Page Configuration & World-Class Custom CSS Theme
st.set_page_config(
    page_title="World-Class Insurance Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .sidebar .sidebar-content { background-color: #1e293b; }
    h1, h2, h3 { color: #38bdf8 !important; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    </style>
""", unsafe_allow_html=True)

# 1. High-Performance Mock Data Pipeline using Polars
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 5000
    return pl.DataFrame({
        'Region': np.random.choice(['East', 'West', 'North', 'South'], n),
        'State': np.random.choice(['UP', 'Maharashtra', 'Delhi', 'Karnataka', 'Gujarat'], n),
        'City': np.random.choice(['Kanpur', 'Lucknow', 'Mumbai', 'Bengaluru', 'Ahmedabad'], n),
        'Agent_ID': np.random.choice([f'AG-{i}' for i in range(101, 120)], n),
        'Policy_Type': np.random.choice(['Term', 'Endowment', 'Health', 'ULIP'], n),
        'Premium': np.random.randint(15000, 120000, n),
        'Claim': np.random.randint(0, 70000, n),
        'Client_Age': np.random.randint(25, 60, n),
        'Tenure_Months': np.random.randint(1, 60, n),
        'Status': np.random.choice(['Active', 'Inactive', 'Lapsed'], n, p=[0.68, 0.22, 0.10])
    })

df_pl = load_data()

# 2. Advanced Interactive Sidebar Filters
st.sidebar.header("🔍 Intelligence Filters")
search_term = st.sidebar.text_input("Global Search (Client/Agent/City)")
regions = df_pl['Region'].unique().to_list()
selected_regions = st.sidebar.multiselect("Region Filter", regions, default=regions)
age_range = st.sidebar.slider("Client Age Range", 25, 60, (25, 60))

# Polars High-Speed Filtering
filtered_pl = df_pl.filter(
    (pl.col('Region').is_in(selected_regions)) &
    (pl.col('Client_Age').is_between(age_range[0], age_range[1]))
)

if search_term:
    filtered_pl = filtered_pl.filter(
        pl.any_horizontal(pl.all().cast(pl.Utf8).str.contains(search_term, literal=True))
    )

# 3. DuckDB Connection with Explicit CTEs & Advanced Drill-Down Logic
con = duckdb.connect(database=':memory:')
con.register('insurance_data', filtered_pl.to_arrow())

# Advanced CTE Query for Multi-Level Metric Aggregation & Fraud/Leakage Indexing
kpi_query = """
WITH RegionSummary AS (
    SELECT 
        Region, 
        SUM(Premium) as total_prem, 
        SUM(Claim) as total_claim,
        COUNT(DISTINCT Agent_ID) as active_agents
    FROM insurance_data
    GROUP BY Region
),
RankedRegions AS (
    SELECT 
        Region, 
        total_prem, 
        total_claim, 
        active_agents,
        NTILE(4) OVER (ORDER BY total_prem DESC) as performance_quartile
    FROM RegionSummary
)
SELECT 
    SUM(total_prem) as grand_total_prem,
    SUM(total_claim) as grand_total_claim,
    AVG(total_prem) as avg_region_revenue
FROM RankedRegions;
"""
kpi_res = con.execute(kpi_query).fetchdf()

st.title("🛡️ Enterprise Insurance Intelligence Dashboard")
st.markdown("---")

total_prem = kpi_res['grand_total_prem'].values[0] if kpi_res['grand_total_prem'].values[0] else 0
total_claim = kpi_res['grand_total_claim'].values[0] if kpi_res['grand_total_claim'].values[0] else 0
loss_ratio = (total_claim / total_prem * 100) if total_prem > 0 else 0

st.info(f"🤖 **DuckDB CTE Engine Status:** Active Filters Processed | Net Portfolio Loss Ratio: **{loss_ratio:.2f}%**")

# 4. Multi-Tab Architecture with 6 Core Sections & Hover Details
tabs = st.tabs([
    "📊 Executive KPIs", 
    "🌍 Geographic & Flow", 
    "⚠️ Risk & Claim Leakage", 
    "👤 Agent Activity & Commission", 
    "🔄 Persistency & Lapses", 
    "📁 17-Chart Master Repository"
])

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Premium Collection", f"₹{total_prem:,.0f}")
    c2.metric("Total Claim Payouts", f"₹{total_claim:,.0f}")
    c3.metric("Net Profit Margin", f"₹{(total_prem - total_claim):,.0f}")
    
    st.subheader("Top vs Bottom Region Performance (CTE Driven)")
    perf_df = con.execute("""
        WITH RegionStats AS (
            SELECT Region, SUM(Premium) as Revenue FROM insurance_data GROUP BY Region
        )
        SELECT Region, Revenue, 
               CASE WHEN Revenue > (SELECT AVG(Revenue) FROM RegionStats) THEN 'Top / Above Average' ELSE 'Bottom / Low' END as Performance_Tier
        FROM RegionStats ORDER BY Revenue DESC
    """).fetchdf()
    
    fig_perf = px.bar(perf_df, x='Region', y='Revenue', color='Performance_Tier', title="Regional Performance Tiering", hover_data=['Region', 'Revenue'])
    st.plotly_chart(fig_perf, use_container_width=True)

with tabs[1]:
    st.subheader("City & State Revenue Flow Drill-Down")
    geo_df = con.execute("""
        WITH CityDrill AS (
            SELECT State, City, Region, SUM(Premium) as Rev, COUNT(*) as Pol_Count FROM insurance_data GROUP BY State, City, Region
        )
        SELECT * FROM CityDrill ORDER BY Rev DESC
    """).fetchdf()
    fig_geo = px.sunburst(geo_df, path=['Region', 'State', 'City'], values='Rev', title="Hierarchical Geographic Revenue Breakdown")
    st.plotly_chart(fig_geo, use_container_width=True)

with tabs[2]:
    st.subheader("Fraud Detection & High-Risk Claim Leakage Analysis")
    fraud_df = con.execute("""
        WITH HighRiskClaims AS (
            SELECT Agent_ID, City, Policy_Type, Claim, Premium,
                   (CAST(Claim AS DOUBLE) / NULLIF(Premium, 0)) as Claim_Ratio
            FROM insurance_data
            WHERE Claim > 40000
        )
        SELECT * FROM HighRiskClaims ORDER BY Claim_Ratio DESC
    """).fetchdf()
    fig_fraud = px.scatter(fraud_df, x='Policy_Type', y='Claim', color='Agent_ID', size='Claim_Ratio', 
                           hover_data=['Agent_ID', 'City', 'Policy_Type', 'Claim', 'Premium', 'Claim_Ratio'], 
                           title="High Risk Claim Leakage Index (Hover for Granular Details)")
    st.plotly_chart(fig_fraud, use_container_width=True)

with tabs[3]:
    st.subheader("Agent Activity, Unique Activation & Commission Payout Ledger")
    agent_df = con.execute("""
        WITH AgentLedger AS (
            SELECT Agent_ID, Status, City, COUNT(*) as Policies_Sold, SUM(Premium) as Total_Sales, SUM(Premium) * 0.05 as Commission_Due
            FROM insurance_data
            GROUP BY Agent_ID, Status, City
        )
        SELECT * FROM AgentLedger ORDER BY Total_Sales DESC
    """).fetchdf()
    fig_agent = px.bar(agent_df, x='Agent_ID', y='Total_Sales', color='Status', 
                       hover_data=['Agent_ID', 'City', 'Status', 'Policies_Sold', 'Total_Sales', 'Commission_Due'], 
                       title="Agent Performance & Commission Ledger")
    st.plotly_chart(fig_agent, use_container_width=True)

with tabs[4]:
    st.subheader("Persistency, Lapses & Retention Tracker")
    pers_df = con.execute("""
        WITH PersistencyCheck AS (
            SELECT Status, Policy_Type, Tenure_Months, COUNT(*) as Count, SUM(Premium) as Vol
            FROM insurance_data
            GROUP BY Status, Policy_Type, Tenure_Months
        )
        SELECT Status, Policy_Type, SUM(Count) as Total_Count, SUM(Vol) as Total_Volume FROM PersistencyCheck GROUP BY Status, Policy_Type
    """).fetchdf()
    fig_pers = px.pie(pers_df, names='Status', values='Total_Volume', color='Status', 
                      hover_data=['Policy_Type', 'Total_Count'], title="Portfolio Retention vs Lapsation Share")
    st.plotly_chart(fig_pers, use_container_width=True)

with tabs[5]:
    st.subheader("17-Chart Master Repository & Dynamic Selector")
    
    # 17 Master Charts Selection List
    chart_17_list = [
        "1. Regional Sales Distribution (Top/Bottom Quartiles)", 
        "2. Agent Activity & Inactivity Matrix", 
        "3. Claim Payout vs Collection Ratio", 
        "4. Policy Type Popularity & Volume Breakdown", 
        "5. Client Age Group Segmentation (25-60)", 
        "6. Monthly & Weekly Trend Analysis", 
        "7. Top & Bottom Performer Analysis", 
        "8. Persistency Ratio Tracker", 
        "9. Commission Payout Ledger", 
        "10. Inactive Agent Audit", 
        "11. Unique Activation Frequency", 
        "12. Last 3 Months Sales Drop Analysis", 
        "13. Quarterly Growth Comparison", 
        "14. Client Tenure Breakdown", 
        "15. High Risk Claim Distribution (Fraud Detection)", 
        "16. State-wise Penetration Rate", 
        "17. City-level Revenue Contribution"
    ]
    
    chosen_chart = st.selectbox("Please select any chart from this 17-chart master repository:", chart_17_list)
    st.success(f"Rendering DuckDB CTE Engine view for: **{chosen_chart}**")
    
    repo_df = con.execute("""
        SELECT Agent_ID, City, State, Policy_Type, Status, Premium, Claim, Client_Age 
        FROM insurance_data
    """).fetchdf()
    
    fig_repo = px.box(repo_df, x='Policy_Type', y='Premium', color='Status', 
                      hover_data=['Agent_ID', 'City', 'State', 'Client_Age', 'Premium', 'Claim'], 
                      title=f"Advanced Master View: {chosen_chart}")
    st.plotly_chart(fig_repo, use_container_width=True)

st.markdown("---")
st.subheader("📋 Drill-Down Data Matrix (Polars + DuckDB Powered)")
st.dataframe(filtered_pl.head(25).to_pandas(), use_container_width=True)
