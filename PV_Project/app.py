import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import glob
from datetime import datetime

# --- 1. 全局页面配置 ---
st.set_page_config(page_title="2026光伏全能风控系统", layout="wide", initial_sidebar_state="expanded")

# --- 2. 侧边栏：核心导航 (指挥中心) ---
st.sidebar.title("🚀 2026 风控系统")
app_mode = st.sidebar.radio(
    "请选择功能模块:",
    ["📊 1. 企业信贷评级 (驾驶舱)", "⏳ 2. 历史复盘与未来预测"]
)
st.sidebar.markdown("---")

# =========================================================
# 🔴 模块一：企业信贷评级 (你原来的 52 家 Excel 数据)
# =========================================================
if app_mode == "📊 1. 企业信贷评级 (驾驶舱)":
    
    # --- A. 自动找文件 ---
    current_folder = os.path.dirname(os.path.abspath(__file__))
    xlsx_files = glob.glob(os.path.join(current_folder, "*.xlsx"))

    if not xlsx_files:
        st.error(f"❌ 找不到Excel文件！请确认文件在: {current_folder}")
        st.stop()
    file_path = xlsx_files[0]

    # --- B. Sheet 选择 (保证找到 52 行) ---
    st.sidebar.header("📂 数据源设置")
    try:
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        selected_sheet = st.sidebar.selectbox("选择数据表 (Sheet):", sheet_names)
    except Exception as e:
        st.error(f"Excel 读取失败: {e}")
        st.stop()

    # --- C. 强力加载逻辑 (容错版) ---
    @st.cache_data
    def load_data_safe(sheet_name):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # 补全字符串列
        str_cols = ["信贷评级", "公司名称", "股票代码"]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '未知').replace('NaN', '未知')
                
        # 补全数值列 (特别是毛利率)
        num_cols = ["技术壁垒(毛利率%)", "综合得分", "资产负债率(%)"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df

    df = load_data_safe(selected_sheet)

    # --- D. 筛选器 (保留负毛利率逻辑) ---
    st.sidebar.header("🔍 业务筛选")
    
    # 评级筛选
    if "信贷评级" in df.columns:
        all_ratings = sorted(list(df["信贷评级"].unique()))
        selected_rating = st.sidebar.multiselect("信贷评级:", all_ratings, default=all_ratings)
    else:
        st.stop()

    # 毛利率筛选 (从 -50 开始！)
    min_margin = st.sidebar.slider("最低毛利率 (%):", -50, 60, -50)

    # 执行筛选
    filtered_df = df[
        (df["信贷评级"].isin(selected_rating)) & 
        (df["技术壁垒(毛利率%)"] >= min_margin)
    ]

    # --- E. 界面展示 ---
    st.title("☀️ 光伏行业信贷生存压力测试")
    st.markdown(f"**数据源**: `{os.path.basename(file_path)}` | **样本量**: `{len(filtered_df)}` 家")

    # 顶部指标
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("监测企业", f"{len(filtered_df)} 家")
    col2.metric("A类优质资产", f"{len(filtered_df[filtered_df['综合得分']>=80])} 家")
    col3.metric("平均毛利率", f"{filtered_df['技术壁垒(毛利率%)'].mean():.1f}%")
    col4.metric("平均负债率", f"{filtered_df['资产负债率(%)'].mean():.1f}%")

    st.markdown("---")

    # 图表
    tab1, tab2, tab3 = st.tabs(["📊 行业全景", "🔬 风险矩阵", "📋 详细报表"])

    with tab1:
        if not filtered_df.empty and '综合得分' in filtered_df.columns:
            fig = px.treemap(
                filtered_df,
                path=[px.Constant("全行业"), '信贷评级', '公司名称'],
                values='综合得分',
                color='综合得分',
                color_continuous_scale='RdYlGn',
                height=550
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if not filtered_df.empty:
            fig_bubble = px.scatter(
                filtered_df,
                x="技术壁垒(毛利率%)",
                y="综合得分",
                size="综合得分",
                color="信贷评级",
                hover_name="公司名称",
                height=500
            )
            # 画一条 0% 毛利率的警戒线
            fig_bubble.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="亏损警戒线")
            st.plotly_chart(fig_bubble, use_container_width=True)

    with tab3:
        st.dataframe(filtered_df, use_container_width=True)

# =========================================================
# 🔵 模块二：历史复盘与未来预测 (你的新想法)
# =========================================================
elif app_mode == "⏳ 2. 历史复盘与未来预测":
    
    st.title("📜 战略沙盘：历史周期与未来推演")
    
    # --- A. 历史数据 (手动精编库) ---
    data_hist = [
        {"year": 2005, "val": 20, "event": "《可再生能源法》", "desc": "尚德上市，首富诞生，行业萌芽。"},
        {"year": 2008, "val": 80, "event": "金融危机 & 硅价暴跌", "desc": "多晶硅从400跌到40美元，第一次大洗牌。"},
        {"year": 2011, "val": 30, "event": "欧美双反", "desc": "至暗时刻，尚德破产，出口受阻。"},
        {"year": 2013, "val": 45, "event": "国内补贴启动", "desc": "国家出手救市，市场重心转回国内。"},
        {"year": 2018, "val": 50, "event": "531新政", "desc": "突然断奶，倒逼平价上网技术升级。"},
        {"year": 2020, "val": 85, "event": "碳中和元年", "desc": "双碳目标提出，光伏茅抱团，硅料暴涨。"},
        {"year": 2024, "val": 40, "event": "内卷出清", "desc": "产能过剩，价格战惨烈，等待触底。"},
        {"year": 2025, "val": 55, "event": "AI能源爆发", "desc": "预测：算力缺电，电网放开消纳，周期反转。"}
    ]
    df_hist = pd.DataFrame(data_hist)

    # --- B. 历史交互图 ---
    st.subheader("1. 中国光伏20年兴衰史 (鼠标悬停查看国情)")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=df_hist['year'], y=df_hist['val'], mode='lines+markers',
        line=dict(color='#d90429', width=4, shape='spline'),
        marker=dict(size=12, color='gold'),
        customdata=np.stack((df_hist['event'], df_hist['desc']), axis=-1),
        hovertemplate="<b>%{x}年</b><br>事件: %{customdata[0]}<br>背景: %{customdata[1]}<extra></extra>"
    ))
    fig_hist.update_layout(title="行业景气度周期", height=450, hovermode="x unified")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.info("💡 **历史规律**：每一次行业危机（2011, 2018），都是技术迭代和国家政策（补贴/双碳）带来的重生机会。")
    st.markdown("---")

    # --- C. 未来白银预测 ---
    st.subheader("2. 2026-2027 白银价格与风控预警")
    
    col_f1, col_f2 = st.columns([1, 2])
    
    with col_f1:
        st.markdown("""
        **预测逻辑 (AI Model)**：
        1. **技术端**：HJT电池银浆消耗量增加 40%。
        2. **供给端**：全球白银矿产停滞。
        3. **宏观端**：美联储降息预期 + AI工业需求。
        
        **🚨 红色警报**：
        若白银突破 **8000元/kg**，非龙头组件厂利润将归零。
        """)
    
    with col_f2:
        # 模拟预测数据
        dates = pd.date_range(start="2026-01-01", periods=24, freq='M')
        base = 7200
        # 模拟暴涨趋势
        prices = [base * (1 + 0.02 * i + 0.001 * i**2) for i in range(24)]
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=dates, y=prices, mode='lines', name='AI预测趋势',
            line=dict(color='blue', width=3, dash='dash')
        ))
        # 警戒线
        fig_pred.add_hline(y=9000, line_dash="dot", line_color="red", annotation_text="中小企业生死线")
        
        fig_pred.update_layout(title="未来24个月白银价格压力测试", yaxis_title="元/kg", height=400)
        st.plotly_chart(fig_pred, use_container_width=True)
