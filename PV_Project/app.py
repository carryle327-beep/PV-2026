import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import glob
from datetime import datetime

# --- 1. 页面配置 ---
st.set_page_config(page_title="SCB光伏风控驾驶舱", layout="wide", initial_sidebar_state="expanded")

# --- 2. 核心美化：注入渣打银行风格 (CSS) ---
# 这段代码会强制覆盖 Streamlit 的原生样式，实现"按键升级"
st.markdown("""
    <style>
    /* 全局背景与字体 */
    .stApp {
        background-color: #F5F7F9; /* 极淡的灰白背景，护眼 */
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* 渣打风格按钮 (Standard Chartered Blue) */
    .stButton>button {
        background-color: #005EBB; /* 渣打蓝 */
        color: white;
        border-radius: 6px; /* 圆角 */
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); /* 微阴影，增加立体感 */
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #004C99; /* 悬停变深 */
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-1px);
    }
    
    /* 侧边栏美化 */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* 标题颜色 (渣打绿/蓝) */
    h1, h2, h3 {
        color: #0B0F32; /* 深蓝黑 */
        font-weight: 700;
    }
    
    /* 指标卡片优化 */
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #009F4D; /* 渣打绿左边框 */
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 核心数据引擎：构建连续历史数据 (2000-2026) ---
@st.cache_data
def get_continuous_history():
    # 1. 定义关键节点 (Anchor Points)
    anchors = {
        2000: {"val": 10, "event": "萌芽期", "desc": "中国光伏产业零星起步，主要做电池片代工。"},
        2005: {"val": 25, "event": "造富神话", "desc": "无锡尚德上市，施正荣成首富，各种热钱涌入。"},
        2008: {"val": 90, "event": "极度过热", "desc": "多晶硅价格冲上400美元天价，拥硅为王。"},
        2009: {"val": 40, "event": "金融危机", "desc": "全球次贷危机爆发，需求骤减，泡沫破裂。"},
        2011: {"val": 20, "event": "欧美双反", "desc": "至暗时刻，尚德破产，全行业亏损。"},
        2013: {"val": 45, "event": "国内启动", "desc": "国家出台电价补贴，市场重心转回国内。"},
        2016: {"val": 70, "event": "领跑者计划", "desc": "单晶替代多晶，技术驱动产业升级。"},
        2018: {"val": 35, "event": "531新政", "desc": "国家断奶，严控规模，行业第二次大洗牌。"},
        2020: {"val": 85, "event": "碳中和元年", "desc": "双碳目标提出，光伏茅抱团，硅料暴涨。"},
        2022: {"val": 100, "event": "俄乌爆发", "desc": "欧洲能源危机，出口井喷，跨界玩家疯狂涌入。"},
        2024: {"val": 30, "event": "产能出清", "desc": "全产业链价格战，跌破成本线，P型产能淘汰。"},
        2026: {"val": 65, "event": "周期反转", "desc": "AI算力缺电 + 供给侧改革完成，新一轮景气周期。"}
    }
    
    # 2. 生成连续年份 (2000 - 2026)
    full_years = list(range(2000, 2027))
    data = []
    
    # 3. 插值算法 (Interpolation) - 填补中间年份
    # 将字典转为Series以便插值
    df_anchors = pd.DataFrame.from_dict(anchors, orient='index').reindex(full_years)
    
    # 线性插值计算 value
    df_anchors['val'] = df_anchors['val'].interpolate(method='linear')
    
    # 填充描述 (非关键年份填“市场自然波动”)
    for year in full_years:
        row = df_anchors.loc[year]
        event = row['event'] if pd.notna(row['event']) else "市场自然演变"
        desc = row['desc'] if pd.notna(row['desc']) else f"{year}年，行业处于周期过渡阶段，技术稳步积累。"
        
        data.append({
            "year": year,
            "index": round(row['val'], 1),
            "event": event,
            "desc": desc
        })
        
    return pd.DataFrame(data)

# --- 4. 侧边栏导航 ---
st.sidebar.markdown("### 🏦 SCB Risk Dashboard")
app_mode = st.sidebar.radio(
    "Select Module / 选择模块:",
    ["📈 1. 行业周期复盘 (History)", "📊 2. 企业信贷评级 (Credit)"]
)
st.sidebar.info("Data Source: SCB Internal & EastMoney")

# =========================================================
# 🔵 模块一：行业周期复盘 (历年连续数据)
# =========================================================
if app_mode == "📈 1. 行业周期复盘 (History)":
    
    st.title("📈 中国光伏产业 26 年全景复盘")
    st.markdown("**(2000 - 2026 连续周期趋势图)**")
    
    # 1. 获取连续数据
    df_hist = get_continuous_history()
    
    # 2. 绘制渣打风格折线图
    fig = go.Figure()

    # 添加区域填充 (Area Chart) - 渣打蓝渐变
    fig.add_trace(go.Scatter(
        x=df_hist['year'], 
        y=df_hist['index'],
        mode='lines+markers',
        name='行业景气指数',
        # 渣打蓝线条
        line=dict(color='#005EBB', width=3, shape='spline'), # spline让线条变圆滑
        # 填充颜色 (浅蓝)
        fill='tozeroy',
        fillcolor='rgba(0, 94, 187, 0.1)',
        # 标记点 (渣打绿)
        marker=dict(
            size=8, 
            color='white', 
            line=dict(width=2, color='#009F4D') # 绿色边框
        ),
        # 悬停交互
        customdata=np.stack((df_hist['event'], df_hist['desc']), axis=-1),
        hovertemplate="<br>".join([
            "<b>📅 %{x}年</b>",
            "📊 景气指数: %{y}",
            "🏷️ <b>%{customdata[0]}</b>",
            "📝 %{customdata[1]}",
            "<extra></extra>"
        ])
    ))

    # 3. 布局美化 (金融终端风格)
    fig.update_layout(
        title="", # 标题在外面写
        xaxis=dict(
            title="Year / 年份", 
            tickmode='linear', 
            dtick=1, # 强制显示每一年！
            showgrid=False,
            tickangle=-45 # 年份斜着放，防止重叠
        ),
        yaxis=dict(
            title="Index / 景气度", 
            showgrid=True, 
            gridcolor='#E5E5E5', # 极淡的网格
            gridwidth=1,
            zeroline=False
        ),
        height=550,
        hovermode="x unified", # 统一悬停线
        plot_bgcolor='white', # 纯白背景
        margin=dict(l=40, r=40, t=20, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # 4. 关键年份详情列表
    st.subheader("📋 关键节点纪要")
    # 只显示有大事件的年份
    key_events = df_hist[df_hist['event'] != "市场自然演变"].sort_values(by="year", ascending=False)
    st.dataframe(
        key_events[['year', 'event', 'desc']], 
        hide_index=True,
        column_config={
            "year": "年份",
            "event": "关键事件",
            "desc": "国情与政策背景"
        },
        use_container_width=True
    )

# =========================================================
# 🔵 模块二：企业信贷评级 (原功能升级版)
# =========================================================
elif app_mode == "📊 2. 企业信贷评级 (Credit)":
    
    # --- 自动加载逻辑 ---
    current_folder = os.path.dirname(os.path.abspath(__file__))
    xlsx_files = glob.glob(os.path.join(current_folder, "*.xlsx"))

    if not xlsx_files:
        st.error(f"❌ System Error: Data file not found in {current_folder}")
        st.stop()
    file_path = xlsx_files[0]

    # Sheet 选择
    try:
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        # 侧边栏选择，更紧凑
        selected_sheet = st.sidebar.selectbox("Select Sheet:", sheet_names)
    except Exception as e:
        st.error(f"Read Error: {e}")
        st.stop()

    # 数据读取与清洗
    @st.cache_data
    def load_data(sheet):
        df = pd.read_excel(file_path, sheet_name=sheet)
        # 强力清洗
        str_cols = ["信贷评级", "公司名称", "股票代码"]
        for c in str_cols:
            if c in df.columns: df[c] = df[c].astype(str).replace(['nan','NaN'], 'N/A')
        
        num_cols = ["技术壁垒(毛利率%)", "综合得分", "资产负债率(%)"]
        for c in num_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df

    df = load_data(selected_sheet)

    # 筛选器
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filters / 筛选")
    
    if "信贷评级" in df.columns:
        all_ratings = sorted(list(df["信贷评级"].unique()))
        selected_rating = st.sidebar.multiselect("Credit Rating:", all_ratings, default=all_ratings)
    else:
        st.error("Missing column: 信贷评级")
        st.stop()
        
    min_margin = st.sidebar.slider("Min Margin (毛利率):", -50, 60, -50)

    filtered_df = df[
        (df["信贷评级"].isin(selected_rating)) & 
        (df["技术壁垒(毛利率%)"] >= min_margin)
    ]

    # --- 仪表盘展示 ---
    st.title("🛡️ 2026 Corporate Credit Stress Test")
    
    # 渣打风格指标卡 (通过CSS美化)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monitor Count", f"{len(filtered_df)}", "Companies")
    c2.metric("Grade A Assets", f"{len(filtered_df[filtered_df['综合得分']>=80])}", "High Quality")
    c3.metric("Avg Margin", f"{filtered_df['技术壁垒(毛利率%)'].mean():.1f}%", "Profitability")
    c4.metric("Avg Debt Ratio", f"{filtered_df['资产负债率(%)'].mean():.1f}%", "Risk Level")

    st.markdown("---")
    
    t1, t2 = st.tabs(["📊 Portfolio View (全景)", "📋 Data Details (明细)"])
    
    with t1:
        # 气泡图：渣打配色
        if not filtered_df.empty:
            fig = px.scatter(
                filtered_df,
                x="技术壁垒(毛利率%)",
                y="综合得分",
                size="综合得分",
                color="信贷评级",
                hover_name="公司名称",
                color_discrete_sequence=px.colors.qualitative.Safe, # 安全色系
                height=500
            )
            # 警戒线
            fig.add_vline(x=0, line_dash="dash", line_color="#D90429", annotation_text="Breakeven Point")
            fig.update_layout(plot_bgcolor="white", xaxis=dict(showgrid=True, gridcolor="#eee"), yaxis=dict(showgrid=True, gridcolor="#eee"))
            st.plotly_chart(fig, use_container_width=True)
            
    with t2:
        st.dataframe(filtered_df, use_container_width=True)
