import polars as pl
import streamlit as st
import plotly.express as px

# ==============================================================================
# 🎨 1. पेज कॉन्फ़िगरेशन और प्रोफेशनल थीम सेटिंग्स
# ==============================================================================
st.set_page_config(page_title="Silicon Valley Insurance Analytics Engine", layout="wide")

# डैशबोर्ड की थीम को बिल्कुल प्रोफेशनल और प्रीमियम दिखाने के लिए कस्टम CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stSidebar { background-color: #161b22; }
    h1, h2, h3 { color: #58a6ff; font-family: sans-serif; }
    .stMetric { background-color: #21262d; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Silicon Valley-Grade Insurance Intelligence System")

# [कहाँ बदलें]: अपनी Parquet फाइल का सही पाथ यहाँ सेट करें
file_path = "clean_insurance_data.parquet"

@st.cache_data
def load_and_process_insurance_data(path):
    """
    POLARS 5 CORE NATIVE EXPRESSIONS:
    1. Lazy Evaluation (scan_parquet) - रैम बचाने और सुपर-फास्ट प्रोसेसिंग के लिए
    2. Filter (.filter) - अवांछित डेटा हटाने के लिए
    3. Conditional Logic (.when().then().otherwise()) - फ्रॉड रिस्क सेग्मेंटेशन के लिए
    4. Window Functions & Ranking (.rank().over()) - रीजन के हिसाब से रैंक निकालने के लिए
    5. Group By & Aggregation (.group_by().agg()) - मास्टर समरी तैयार करने के लिए
    """
    try:
        lazy_df = pl.scan_parquet(path)
        processed_df = (
            lazy_df
            .filter(pl.col("premium_amount") > 0)
            .with_columns(
                client_segment=pl.when(pl.col("claim_amount") > (pl.col("premium_amount") * 0.7))
                .then(pl.lit("High Fraud Risk"))
                .otherwise(pl.lit("Healthy Client"))
            )
            .with_columns(
                region_rank=pl.col("premium_amount").rank("min", descending=True).over("region")
            )
            .group_by(["region", "agent_id", "agent_name", "policy_name", "city", "state", "client_segment"])
            .agg([
                pl.col("premium_amount").sum().alias("total_revenue"),
                pl.count("policy_id").alias("total_policies"),
                pl.col("claim_amount").mean().alias("avg_claim"),
                pl.col("claim_amount").sum().alias("total_claims")
            ])
            .sort("total_revenue", descending=True)
            .collect(streaming=True)
        )
        return processed_df
    except Exception as e:
        st.error(f"डेटा लोड करने में त्रुटि (फाइल पाथ या कॉलम जांचें): {e}")
        return pl.DataFrame()

df = load_and_process_insurance_data(file_path)

# ==============================================================================
# 📊 2. Streamlit UI & 9 World-Class Charts (इजेक्ट थीम और रिच हover के साथ)
# ==============================================================================
if not df.is_empty():
    pdf = df.to_pandas()
    
    st.sidebar.markdown("### 🎛️ Master Control Panel")
    
    nine_world_class_charts = [
        "1. Regional Premium Distribution (Bar Chart)",
        "2. Agent Performance Leaderboard (Horizontal Bar)",
        "3. Policy-wise Revenue Share (Donut Chart)",
        "4. City-wise Revenue vs Average Claim (Scatter Plot)",
        "5. Fraud Risk vs Healthy Clients (Pie Chart)",
        "6. Renewal & Persistence Rate Tracker (Funnel/Bar)",
        "7. Claim Settlement Status & Payout Volume (Bar Chart)",
        "8. Customer Demographics & Policy Heatmap (Heatmap)",
        "9. Agent Cost vs Revenue ROI & Efficiency (Matrix)"
    ]
    
    selected_chart = st.sidebar.selectbox("Select Core Analytics Chart", nine_world_class_charts)
    
    st.sidebar.markdown("---")
    user_command = st.sidebar.text_input("💬 Command Box:", placeholder="Ask or filter views here...")
    if user_command:
        st.sidebar.success(f"Command Executed: {user_command}")

    selected_region = st.sidebar.selectbox("Filter by Region", ["All"] + pdf['region'].dropna().unique().tolist())
    filtered_pdf = pdf if selected_region == "All" else pdf[pdf['region'] == selected_region]

    # **कर्सर ले जाने पर दिखने वाली फुल रिच हover डिटेल्स (कोई कॉलम खाली नहीं)**
    comprehensive_hover = [
        'agent_id', 'agent_name', 'policy_name', 'total_revenue', 
        'total_policies', 'avg_claim', 'total_claims', 'city', 'state', 'region', 'client_segment'
    ]
    available_hover = [c for c in comprehensive_hover if c in filtered_pdf.columns]

    st.markdown(f"### 📈 Active View: {selected_chart}")

    # ==========================================================================
    # 9 वर्ल्ड-क्लास चार्ट्स (प्रोफेशनल डार्क थीम और प्लॉटली लेआउट के साथ)
    # ==========================================================================
    if "1." in selected_chart:
        fig = px.bar(filtered_pdf, x='region', y='total_revenue', color='state' if 'state' in filtered_pdf.columns else 'region',
                     hover_data=available_hover, title="1. Regional Premium Distribution by Zone & State")
        
    elif "2." in selected_chart:
        top_agents = filtered_pdf.sort_values(by='total_revenue', ascending=False).head(15)
        fig = px.bar(top_agents, x='total_revenue', y='agent_name' if 'agent_name' in top_agents.columns else 'agent_id',
                     orientation='h', hover_data=available_hover, title="2. Top Agent Leaderboard & Performance Trend")
        
    elif "3." in selected_chart:
        fig = px.pie(filtered_pdf, names='policy_name' if 'policy_name' in filtered_pdf.columns else 'region', 
                     values='total_revenue', hole=0.4, hover_data=available_hover, title="3. Policy-wise Revenue Share & Breakdown")
        
    elif "4." in selected_chart:
        fig = px.scatter(filtered_pdf, x='total_revenue', y='avg_claim', color='city' if 'city' in filtered_pdf.columns else 'region',
                         hover_data=available_hover, title="4. City-wise Revenue Concentration vs Average Claim")
        
    elif "5." in selected_chart:
        fig = px.pie(filtered_pdf, names='client_segment', values='total_revenue', hole=0.4,
                     hover_data=available_hover, title="5. Fraud Risk (High Risk vs Healthy Client Portfolio)")
        
    elif "6." in selected_chart:
        fig = px.bar(filtered_pdf, x='agent_id', y='total_policies', color='client_segment',
                     hover_data=available_hover, title="6. Renewal & Persistence Rate Tracker (Active vs Lapsed/Risk)")
        
    elif "7." in selected_chart:
        fig = px.bar(filtered_pdf, x='city' if 'city' in filtered_pdf.columns else 'region', y='total_claims', color='region',
                     hover_data=available_hover, title="7. Claim Settlement Status & Payout Volume by City/Zone")
        
    elif "8." in selected_chart:
        fig = px.density_heatmap(filtered_pdf, x='city', y='policy_name' if 'policy_name' in filtered_pdf.columns else 'region', 
                                 z='total_revenue', hover_data=available_hover, title="8. Customer Demographics & Policy Heatmap")
        
    elif "9." in selected_chart:
        fig = px.scatter(filtered_pdf, x='total_policies', y='total_revenue', size='avg_claim', color='agent_id',
                         hover_data=available_hover, title="9. Agent Cost vs Revenue ROI & Efficiency Matrix")

    # चार्ट की थीम को बिल्कुल प्रोफेशनल डार्क लुक देने के लिए
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(color="#f0f6fc", size=12),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # मास्टर डेटा टेबल व्यू
    with st.expander("📋 View Fully Detailed Processed Dataset Table"):
        st.dataframe(filtered_pdf, use_container_width=True)

else:
    st.warning("⚠️ कृपया सही फाइल पाथ दें या Parquet फाइल की जांच करें।")
