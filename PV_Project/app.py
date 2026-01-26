import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import glob

# --- 1. 页面配置 ---
st.set_page_config(page_title="SCB光伏风控驾驶舱", layout="wide", initial_sidebar_state="expanded")

# --- 2. 渣打风格 CSS (严格修复版) ---
st.markdown("""
    <style>
    /* 全局背景：极浅的商务灰 */
    .stApp {
        background-color: #F2F5F8;
    }
    
    /* 所有字体强制变深，防止看不见 */
    h1, h2, h3, h4, h5, h6, p, li, span {
        color: #0B0F32 !important; /* 深蓝黑 */
        font-family: 'Arial', sans-serif;
    }
    
    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF; /* 纯白侧边栏 */
        border-right: 1px solid #DDE1E6;
    }
    
    /* 按钮样式：扁平、高级、渣打蓝 */
    .stButton>button {
        background-color: #005EBB !important; /* 渣打蓝 */
        color: white !important;
        border-radius: 4px; /* 稍微方一点，更商务 */
        border: none;
        padding: 8px 20px;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #004C99 !important; /* 深一点的蓝 */
        border: none;
    }
    
    /* 去掉讨厌的红色警告框，改成中性蓝 */
    .stAlert {
        background-color: #E6F0FA;
        border-left-color: #005EBB;
        color: #0B0F32;
    }
    
    /* 指标卡片 (Metric) */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 6px;
        border: 1px solid #E0E0E0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    /* 指标数值颜色：强制改为渣打绿 */
    div[data-testid="stMetricValue"] {
        color: #009F4D !important; /* 渣打绿 */
    }
    
    /* 调整 Tab 样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F2F5F8;
        border-radius: 4px;
        color: #0B0F32;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #005EBB !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 核心数据引擎 (2000-2026 连续年份) ---
@st.cache_data
def get_continuous_history():
    # 关键节点
    anchors = {
        2000: {"val": 15, "event": "起步期", "desc": "零星代工，技术积累。"},
        2005: {"val": 30, "event": "尚德上市", "desc": "造富效应，资本涌入。"},
        2008: {"val": 95, "event": "拥硅为王", "desc": "多晶硅天价泡沫。"},
        2009: {"val": 40, "event": "金融危机", "desc": "泡沫破裂，需求骤降。"},
        2011: {"val": 20, "event": "欧美双反", "desc": "至暗时刻，全行业亏损。"},
        2013: {"val": 50, "event": "国内补贴", "desc": "政策救市，内需启动。"},
        2016: {"val": 75, "event": "领跑者", "desc": "技术升级，单晶替代。"},
        2018: {"val": 35, "event": "531新政", "desc": "断奶去补贴，行业洗牌。"},
        2020: {"val": 85, "event": "碳中和", "desc": "双碳目标，估值重构。"},
        2022: {"val": 100, "event": "欧洲危机", "desc": "俄乌冲突，出口井喷。"},
        2024: {"val": 30, "event": "产能出清", "desc": "价格战底，剩者为王。"},
        2026: {"val": 70, "event": "新周期", "desc": "AI算力缺电，需求反转。"}
    }
    
    full_years = list(range(2000, 2027))
    data = []
    
    df_anchors = pd.DataFrame.from_dict(anchors, orient='index').reindex(full_years)
    df_anchors['val'] = df_anchors['val'].interpolate(method='linear') # 插值
    
    for year in full_years:
        row = df_anchors.loc[year]
        event = row['event'] if pd.notna(row['event']) else "-"
        desc = row['desc'] if pd.notna(row['desc']) else "行业平稳发展期"
        data.append({"year": year, "index": round(row['val'], 1), "event": event, "desc": desc})
        
    return pd.DataFrame(data)

# --- 4. 导航 ---
st.sidebar.title("🏦 SCB Risk Pilot")
st.sidebar.info("Standard Chartered Bank Style")
app_mode = st.sidebar.radio("Module / 模块:", ["📈 行业历史周期 (History)", "📊 企业信贷评级 (Credit)"])

# =========================================================
# 模块一：行业历史周期 (History) - 严禁红色
# =========================================================
if app_mode == "📈 行业历史周期 (History)":
    st.header("📈 中国光伏产业 26 年全景复盘 (2000-2026)")
    
    df_hist = get_continuous_history()
    
    fig = go.Figure()

    # 折线图：渣打蓝 (#005EBB) + 区域填充
    fig.add_trace(go.Scatter(
        x=df_hist['year'], 
        y=df_hist['index'],
        mode='lines+markers',
        name='景气指数',
        line=dict(color='#005EBB', width=3, shape='spline'), # 渣打蓝
        fill='tozeroy',
        fillcolor='rgba(0, 94, 187, 0.08)', # 极淡的蓝色填充
        marker=dict(size=6, color='white', line=dict(width=2, color='#009F4D')), # 渣打绿的生长点
        hovertemplate="<b>%{x}年</b><br>指数: %{y}<br>事件: %{customdata[0]}<br>背景: %{customdata[1]}<extra></extra>",
        customdata=np.stack((df_hist['event'], df_hist['desc']), axis=-1)
    ))

    # 布局：极简商务
    fig.update_layout(
        xaxis=dict(title="Year", tickmode='linear', dtick=1, showgrid=False, tickangle=-45),
        yaxis=dict(title="Index", showgrid=True, gridcolor='#EEEEEE', zeroline=False), # 淡灰网格
        height=500,
        plot_bgcolor='white',
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # 下方展示关键事件表 (只显示有大事的年份)
    st.subheader("📋 关键历史节点")
    key_events = df_hist[df_hist['event'] != "-"].sort_values("year", ascending=False)
    st.dataframe(
        key_events[['year', 'event', 'desc']], 
        hide_index=True,
        use_container_width=True,
        column_config={"year": "年份", "event": "关键事件", "desc": "背景描述"}
    )

# =========================================================
# 模块二：企业信贷评级 (Credit) - 严禁红色
# =========================================================
elif app_mode == "📊 企业信贷评级 (Credit)":
    
    # 自动加载
    current_folder = os.path.dirname(os.path.abspath(__file__))
    xlsx_files = glob.glob(os.path.join(current_folder, "*.xlsx"))
    
    if not xlsx_files:
        st.error("Data File Missing")
        st.stop()
        
    file_path = xlsx_files[0]
    
    # Sheet 选择
    try:
        sheet_names = pd.ExcelFile(file_path).sheet_names
        selected_sheet = st.sidebar.selectbox("Select Sheet:", sheet_names)
    except:
        st.stop()
        
    @st.cache_data
    def load_data(s):
        df = pd.read_excel(file_path, sheet_name=s)
        # 清洗
        for c in df.columns:
            if df[c].dtype == 'object': df[c] = df[c].fillna("-").astype(str)
            else: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df

    df = load_data(selected_sheet)
    
    # 筛选
    st.sidebar.markdown("---")
    if "信贷评级" in df.columns:
        opts = sorted(list(df["信贷评级"].unique()))
        sel = st.sidebar.multiselect("Rating:", opts, default=opts)
    else:
        st.stop()
        
    min_margin = st.sidebar.slider("Min Margin (%):", -50, 60, -50)
    
    filtered_df = df[
        (df["信贷评级"].isin(sel)) & 
        (df["技术壁垒(毛利率%)"] >= min_margin)
    ]
    
    st.header("🛡️ Corporate Credit Stress Test")
    
    # 指标卡
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies", f"{len(filtered_df)}")
    c2.metric("Grade A", f"{len(filtered_df[filtered_df['综合得分']>=80])}")
    c3.metric("Avg Margin", f"{filtered_df['技术壁垒(毛利率%)'].mean():.1f}%")
    c4.metric("Avg Debt", f"{filtered_df['资产负债率(%)'].mean():.1f}%")
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📊 Overview", "📋 Details"])
    
    with tab1:
        if not filtered_df.empty:
            # 气泡图颜色：使用安全的蓝/绿/灰，不用红
            # 手动定义颜色映射，防止 Plotly 自动用红色
            fig = px.scatter(
                filtered_df,
                x="技术壁垒(毛利率%)",
                y="综合得分",
                size="综合得分",
                color="信贷评级",
                hover_name="公司名称",
                height=500,
                color_discrete_sequence=["#005EBB", "#009F4D", "#66CCFF", "#999999", "#FF9900"] # 蓝, 绿, 浅蓝, 灰, 橙(警告)
            )
            # 盈亏平衡线：用橙色虚线代替红色实线
            fig.add_vline(x=0, line_dash="dash", line_color="#FFA500", annotation_text="Breakeven")
            fig.update_layout(plot_bgcolor="white", xaxis=dict(showgrid=True, gridcolor="#eee"), yaxis=dict(showgrid=True, gridcolor="#eee"))
            st.plotly_chart(fig, use_container_width=True)
            
    with tab2:
        st.dataframe(filtered_df, use_container_width=True)
