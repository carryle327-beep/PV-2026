
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import glob

# --- 1. 页面配置 (强制宽屏) ---
st.set_page_config(page_title="SCB Risk Pilot", layout="wide", initial_sidebar_state="expanded")

# --- 2. 极致的高级感 CSS (机构灰风格) ---
st.markdown("""
    <style>
    /* 1. 全局强制重置：覆盖所有 Streamlit 默认颜色 */
    .stApp {
        background-color: #F0F2F5 !important; /* 页面背景：高级冷灰 */
    }
    
    /* 2. 字体优化：模拟金融终端的冷静感 */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', 'Arial', sans-serif !important;
        color: #1A1A1A !important; /* 近似纯黑的深灰，比纯黑更护眼 */
    }
    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.5px !important; /* 紧凑字间距，显得高级 */
        color: #0E1117 !important;
    }
    
    /* 3. 侧边栏：纯白高亮 */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E6E6E6;
        box-shadow: 1px 0 5px rgba(0,0,0,0.02);
    }
    
    /* 4. 按钮：深岩石灰 (彻底告别橙色/蓝色) */
    .stButton>button {
        background-color: #2E3B4E !important; /* 深岩灰：代表理性和稳重 */
        color: #FFFFFF !important;
        border: none;
        border-radius: 2px !important; /* 只有2px圆角，非常硬朗 */
        padding: 10px 24px;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase; /* 英文大写，增加气势 */
        letter-spacing: 1px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1A2533 !important; /* 悬停变黑 */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* 5. 指标卡片：极简纯白块 */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 4px;
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    div[data-testid="stMetricLabel"] {
        color: #666666 !important; /* 标签灰色 */
        font-size: 14px !important;
    }
    div[data-testid="stMetricValue"] {
        color: #0E1117 !important; /* 数值深黑色 */
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    /* 6. 去除所有杂色装饰 */
    .stAlert {
        background-color: #F8F9FA;
        border: 1px solid #E0E0E0;
        color: #333;
    }
    
    /* 7. Tab 页签样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        color: #666;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E3B4E !important; /* 选中变深灰 */
        color: white !important;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 数据生成 (连续历史) ---
@st.cache_data
def get_continuous_history():
    anchors = {
        2000: 15, 2005: 30, 2008: 95, 2009: 40, 
        2011: 20, 2013: 50, 2016: 75, 2018: 35, 
        2020: 85, 2022: 100, 2024: 30, 2026: 70
    }
    events = {
        2000: "起步期", 2005: "尚德上市", 2008: "拥硅为王", 2009: "金融危机",
        2011: "欧美双反", 2013: "国内补贴", 2016: "领跑者", 2018: "531新政",
        2020: "碳中和", 2022: "俄乌冲突", 2024: "产能出清", 2026: "AI新周期"
    }
    
    full_years = list(range(2000, 2027))
    df = pd.DataFrame(index=full_years)
    df['index'] = pd.Series(anchors).reindex(full_years).interpolate(method='linear')
    df['event'] = pd.Series(events).reindex(full_years).fillna("-")
    df = df.reset_index().rename(columns={'index': 'year', 0: 'val'}) # 修正列名
    return df

# --- 4. 侧边栏 (纯净版) ---
st.sidebar.markdown("## 🏛️ SCB RISK PILOT")
st.sidebar.caption("INSTITUTIONAL CLIENTS GROUP")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("MODULE SELECTOR", ["📈 MACRO CYCLE (历史周期)", "📊 CREDIT RATING (信贷评级)"])

# =========================================================
# 模块一：历史周期 (冷色调)
# =========================================================
if app_mode == "📈 MACRO CYCLE (历史周期)":
    st.markdown("### PV INDUSTRY CYCLE: 2000 - 2026")
    st.caption("Historical Trend & Future Projection")
    
    df_hist = get_continuous_history()
    
    fig = go.Figure()

    # 线条：深岩灰 (#2E3B4E) 
    fig.add_trace(go.Scatter(
        x=df_hist['year'], 
        y=df_hist['val'],
        mode='lines+markers',
        name='Index',
        line=dict(color='#2E3B4E', width=3), # 深灰线
        fill='tozeroy',
        fillcolor='rgba(46, 59, 78, 0.1)', # 极淡的灰色填充
        marker=dict(size=6, color='white', line=dict(width=2, color='#2E3B4E')), # 深灰点
        hovertemplate="<b>%{x}</b><br>Index: %{y:.1f}<br>Event: %{customdata}<extra></extra>",
        customdata=df_hist['event']
    ))

    # 布局：极致简约
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=1, showgrid=False, tickangle=-90, color='#666'),
        yaxis=dict(showgrid=True, gridcolor='#E0E0E0', zeroline=False, color='#666'),
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=30, b=20),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # 关键事件表 (纯白背景)
    st.markdown("#### KEY HISTORICAL EVENTS")
    key_events = df_hist[df_hist['event'] != "-"].sort_values("year", ascending=False)
    st.dataframe(
        key_events[['year', 'event']], 
        hide_index=True,
        use_container_width=True,
        column_config={"year": "Year", "event": "Milestone"}
    )

# =========================================================
# 模块二：信贷评级 (冷色调)
# =========================================================
elif app_mode == "📊 CREDIT RATING (信贷评级)":
    
    # 自动加载
    current_folder = os.path.dirname(os.path.abspath(__file__))
    xlsx_files = glob.glob(os.path.join(current_folder, "*.xlsx"))
    
    if not xlsx_files:
        st.error("SYSTEM ERROR: Data file not found.")
        st.stop()
    
    file_path = xlsx_files[0]
    
    try:
        sheet_names = pd.ExcelFile(file_path).sheet_names
        selected_sheet = st.sidebar.selectbox("DATA SHEET", sheet_names)
    except:
        st.stop()
        
    @st.cache_data
    def load_data(s):
        df = pd.read_excel(file_path, sheet_name=s)
        for c in df.columns:
            if df[c].dtype == 'object': df[c] = df[c].fillna("-").astype(str)
            else: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df

    df = load_data(selected_sheet)
    
    # 筛选器
    if "信贷评级" in df.columns:
        opts = sorted(list(df["信贷评级"].unique()))
        sel = st.sidebar.multiselect("RATING FILTER", opts, default=opts)
    else:
        st.stop()
        
    min_margin = st.sidebar.slider("MARGIN THRESHOLD (%)", -50, 60, -50)
    
    filtered_df = df[
        (df["信贷评级"].isin(sel)) & 
        (df["技术壁垒(毛利率%)"] >= min_margin)
    ]
    
    st.markdown("### CORPORATE CREDIT STRESS TEST")
    st.caption(f"Source: {selected_sheet} | Companies: {len(filtered_df)}")
    
    # 指标卡
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("COVERAGE", f"{len(filtered_df)}")
    c2.metric("GRADE A", f"{len(filtered_df[filtered_df['综合得分']>=80])}")
    c3.metric("AVG MARGIN", f"{filtered_df['技术壁垒(毛利率%)'].mean():.1f}%")
    c4.metric("AVG DEBT", f"{filtered_df['资产负债率(%)'].mean():.1f}%")
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["SCATTER PLOT", "DATA GRID"])
    
    with tab1:
        if not filtered_df.empty:
            # 颜色映射：强制使用冷色系 (蓝、灰、青)
            # 绝对不使用红色/橙色
            fig = px.scatter(
                filtered_df,
                x="技术壁垒(毛利率%)",
                y="综合得分",
                size="综合得分",
                color="信贷评级",
                hover_name="公司名称",
                height=500,
                color_discrete_sequence=["#2E3B4E", "#5D6D7E", "#85929E", "#AED6F1", "#3498DB"] # 深灰到浅蓝
            )
            
            # 盈亏平衡线：改为深黑色虚线，极度克制
            fig.add_vline(x=0, line_dash="dot", line_color="#333333", annotation_text="BREAKEVEN")
            
            fig.update_layout(
                plot_bgcolor="white", 
                xaxis=dict(showgrid=True, gridcolor="#F0F0F0", title="Gross Margin (%)"), 
                yaxis=dict(showgrid=True, gridcolor="#F0F0F0", title="Composite Score")
            )
            st.plotly_chart(fig, use_container_width=True)
            
    with tab2:
        st.dataframe(filtered_df, use_container_width=True)
