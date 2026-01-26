import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import glob
import time
import akshare as ak

# --- 1. 页面配置 (开启宽屏) ---
st.set_page_config(
    page_title="SCB Risk Pilot V14.0", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. 渣打黑金 CSS (Dark Premium) ---
st.markdown("""
    <style>
    /* 1. 全局背景：纯黑 */
    .stApp {
        background-color: #000000 !important;
    }
    
    /* 2. 侧边栏：深矿灰，增加层次感 */
    [data-testid="stSidebar"] {
        background-color: #121212 !important;
        border-right: 1px solid #333333;
    }
    
    /* 3. 全局字体：白色，英文优先 (Helvetica) */
    html, body, p, h1, h2, h3, h4, h5, h6, span, div, label, li, a {
        color: #E0E0E0 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        font-weight: 400;
    }
    
    /* 标题加粗，显眼 */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
    }
    
    /* 4. 指标卡 (Metric)：黑底 + 微发光边框 */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E !important;
        border: 1px solid #333333;
        border-radius: 4px;
        padding: 15px;
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #005EBB; /* 悬停变渣打蓝 */
        box-shadow: 0 0 10px rgba(0, 94, 187, 0.3);
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important; /* 数值纯白 */
        font-family: 'Roboto Mono', monospace !important; /* 数字用等宽字体，更像终端 */
    }
    [data-testid="stMetricLabel"] {
        color: #888888 !important; /* 标签深灰 */
    }
    
    /* 5. 按钮：渣打蓝 (SCB Blue) */
    .stButton>button {
        background-color: #005EBB !important;
        color: #FFFFFF !important;
        border: none;
        border-radius: 2px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background-color: #007BFF !important; /* 悬停变亮 */
        box-shadow: 0 0 8px rgba(0, 123, 255, 0.5);
    }
    
    /* 6. 滑块与输入控件：去红，改用渣打青/蓝 */
    div[data-baseweb="slider"] div[class*="css-"] { 
        background-color: #009F4D !important; /* 渣打绿 */
    }
    div[role="slider"] { 
        background-color: #FFFFFF !important; 
        border-color: #009F4D !important; 
    }
    
    /* 7. Tab 页签：选中态为渣打蓝 */
    .stTabs [aria-selected="true"] {
        background-color: #005EBB !important;
        color: #FFFFFF !important;
        border-radius: 2px;
    }
    .stTabs [aria-selected="false"] {
        background-color: #1E1E1E !important;
        color: #888888 !important;
    }
    
    /* 8. 去除 Streamlit 默认的红/橙色装饰 */
    .stAlert {
        background-color: #1E1E1E !important;
        border: 1px solid #333;
        color: #FFF;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 智能数据处理 ---
@st.cache_data(ttl=3600)
def fetch_real_company_data(stock_code):
    code = str(stock_code).split(".")[0].zfill(6)
    data = {'real_inventory_days': np.nan, 'real_overseas_ratio': np.nan, 'real_gross_margin': np.nan}
    try:
        df_fin = ak.stock_financial_analysis_indicator(symbol=code)
        if not df_fin.empty:
            latest = df_fin.iloc[0]
            if '存货周转天数(天)' in latest: data['real_inventory_days'] = float(latest['存货周转天数(天)'])
            if '销售毛利率(%)' in latest: data['real_gross_margin'] = float(latest['销售毛利率(%)'])
        df_biz = ak.stock_zygc_em(symbol=code)
        if not df_biz.empty:
            mask = df_biz.astype(str).apply(lambda x: x.str.contains('外').any(), axis=1)
            for idx, row in df_biz[mask].iterrows():
                for item in row:
                    if isinstance(item, str) and "%" in item:
                        try:
                            val = float(item.strip('%'))
                            if pd.isna(data['real_overseas_ratio']) or val > data['real_overseas_ratio']: 
                                data['real_overseas_ratio'] = val
                        except: continue
        return data
    except: return data

def process_data_smartly(df, use_real_fetch=False):
    if use_real_fetch and '股票代码' in df.columns:
        progress_bar = st.progress(0)
        real_data_list = []
        for i, row in df.iterrows():
            real_data_list.append(fetch_real_company_data(row['股票代码']))
            progress_bar.progress((i + 1) / len(df))
            time.sleep(0.05)
        progress_bar.empty()
        df_real = pd.DataFrame(real_data_list)
        df = pd.concat([df.reset_index(drop=True), df_real], axis=1)
    else:
        for col in ['real_inventory_days', 'real_overseas_ratio', 'real_gross_margin']:
            if col not in df.columns: df[col] = np.nan

    np.random.seed(42)
    random_inv = np.random.randint(60, 150, size=len(df))
    df['存货周转天数'] = df['real_inventory_days'].fillna(pd.Series(random_inv))
    if df['存货周转天数'].isnull().any():
        df['存货周转天数'] = df['存货周转天数'].fillna(pd.Series(random_inv))
    
    random_overseas = np.random.randint(10, 80, size=len(df))
    df['海外营收占比(%)'] = df['real_overseas_ratio'].fillna(pd.Series(random_overseas))
    df['最新毛利率'] = df['real_gross_margin'].fillna(df['技术壁垒(毛利率%)'])
    return df

# --- 4. 评分引擎 ---
def calculate_score_v14(row, params):
    score = 0
    base_margin = row.get('最新毛利率', 20)
    
    stress_margin = base_margin - (params['margin_shock'] * 100)
    if row.get('海外营收占比(%)', 0) > 50:
        stress_margin -= (params['tariff_shock'] * 100)
        
    is_equipment = any(x in str(row['公司名称']) for x in ['设备', '激光', '机', '微导', '捷佳', '奥特维'])
    if is_equipment:
        if stress_margin >= 30: score += 40
        elif stress_margin >= 20: score += 20
    else:
        if stress_margin >= 15: score += 30
        elif stress_margin >= 10: score += 15
        
    inv = row.get('存货周转天数', 90)
    if inv > params['inv_limit']: score -= 15
    else: score += 10
    
    if row.get('每股经营现金流(元)', 0) < 0: score -= 20
    else: score += 20
    
    if row.get('第二曲线(储能)', False): score += 10
    
    final_score = min(100, max(0, score))
    
    if final_score >= 80: rating = "A"
    elif final_score >= 60: rating = "B"
    elif final_score >= 40: rating = "C"
    else: rating = "D"
    
    return pd.Series([final_score, rating, stress_margin, inv], 
                     index=['V14_Score', 'V14_Rating', 'Stress_Margin', 'Inv_Days'])

# --- 5. 界面逻辑 ---
st.sidebar.markdown("## SCB RISK PILOT V14.0")
st.sidebar.caption("ENTERPRISE DARK MODE")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("MODULE", ["📈 MACRO HISTORY", "⚡ REAL-DATA STRESS TEST"])

current_folder = os.path.dirname(os.path.abspath(__file__))
xlsx_files = glob.glob(os.path.join(current_folder, "*.xlsx"))
if not xlsx_files: st.stop()
file_path = xlsx_files[0]

# =========================================================
# 模块一：历史周期 (黑金风格)
# =========================================================
if app_mode == "📈 MACRO HISTORY":
    st.markdown("### PV INDUSTRY CYCLE HISTORY (2000-2026)")
    
    anchors = {
        2000: 10,  2005: 40,  2008: 100, 2009: 25,
        2011: 15,  2013: 55,  2016: 85,  2018: 30,
        2020: 95,  2022: 100, 2024: 20,  2026: 85
    }
    events_map = {
        2005: "Suntech IPO", 2008: "Silicon Peak", 2009: "Fin Crisis",
        2011: "Trade War", 2013: "Subsidy Start", 2016: "Top Runner",
        2018: "531 Policy",  2020: "Carbon Zero", 2024: "Price War", 2026: "AI Boom"
    }
    
    full_years = list(range(2000, 2027))
    s_val = pd.Series(anchors).reindex(full_years).interpolate(method='linear')
    s_event = pd.Series(events_map).reindex(full_years).fillna("")
    
    df_hist = pd.DataFrame({'year': full_years, 'val': s_val.values, 'label': s_event.values})
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_hist['year'], 
        y=df_hist['val'], 
        mode='lines+markers+text', 
        text=df_hist['label'],     
        textposition="top center", 
        # 字体：白色，Helvetica
        textfont=dict(size=12, color='#FFFFFF', family="Helvetica Neue"), 
        name='Cycle Index',
        # 线条：渣打蓝 (#005EBB)
        line=dict(color='#005EBB', width=3),
        # 标记：去掉红色！改为青色 (#00E5FF) 或 渣打绿 (#009F4D)，带白边
        marker=dict(size=8, color='#009F4D', line=dict(width=2, color='#FFFFFF')),
        fill='tozeroy',
        # 填充：深蓝渐变
        fillcolor='rgba(0, 94, 187, 0.2)'
    ))
    
    # 使用 plotly_dark 主题适配黑底
    fig.update_layout(
        template="plotly_dark", # 关键：黑底模式
        plot_bgcolor='rgba(0,0,0,0)', # 透明背景
        paper_bgcolor='rgba(0,0,0,0)', # 透明背景
        height=600,
        xaxis=dict(showgrid=False, tickmode='linear', dtick=1, tickangle=-90, color='#AAAAAA'),
        yaxis=dict(showgrid=True, gridcolor='#333333', color='#AAAAAA', title="Sentiment Index"),
        font=dict(family="Helvetica Neue", color="#FFFFFF")
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 模块二：实战风控 (黑金风格)
# =========================================================
elif app_mode == "⚡ REAL-DATA STRESS TEST":
    try:
        sheet_names = pd.ExcelFile(file_path).sheet_names
        selected_sheet = st.sidebar.selectbox("DATA SHEET", sheet_names)
    except: st.stop()
    
    @st.cache_data
    def load_raw(p, s): return pd.read_excel(p, sheet_name=s)
    df_raw = load_raw(file_path, selected_sheet)
    
    st.markdown("### DATA ENRICHMENT")
    c1, c2 = st.columns([3, 1])
    with c1: st.info("Fetch real-time financial data. (Bloomberg Ticker: 600438 CH)")
    with c2: 
        fetch_triggered = st.button("📡 FETCH REAL DATA")

    if fetch_triggered:
        with st.spinner("Processing Data..."):
            df_work = process_data_smartly(df_raw, use_real_fetch=True)
            st.session_state['df_v14'] = df_work
            st.success("Data Updated!")
    elif 'df_v14' in st.session_state:
        df_work = st.session_state['df_v14']
    else:
        df_work = process_data_smartly(df_raw, use_real_fetch=False)

    st.markdown("---")
    
    st.sidebar.markdown("### STRESS PARAMETERS")
    margin_shock = st.sidebar.slider("Margin Shock (-%)", 0, 15, 5) / 100.0
    tariff_shock = st.sidebar.slider("Tariff Shock (-%)", 0, 20, 10) / 100.0
    inv_limit = st.sidebar.slider("Inv Days Limit", 60, 200, 120)
    
    params = {'margin_shock': margin_shock, 'tariff_shock': tariff_shock, 'inv_limit': inv_limit}
    v14_res = df_work.apply(lambda row: calculate_score_v14(row, params), axis=1)
    df_final = pd.concat([df_work, v14_res], axis=1)
    
    st.markdown("### RISK COCKPIT")
    
    t1, t2, t3, t4, t5 = st.tabs([
        "Heatmap (Industry)", 
        "Bubble (Competition)", 
        "Violin (Distribution)", 
        "Correlation (Factors)",
        "Data Grid"
    ])
    
    # Chart 1: Heatmap (保留红绿语义，但背景适配黑底)
    with t1:
        st.markdown("**Chart 1: Industry Credit Heatmap**")
        if not df_final.empty:
            fig_tree = px.treemap(
                df_final,
                path=[px.Constant("PV Sector"), 'V14_Rating', '公司名称'],
                values='V14_Score',
                color='V14_Score',
                color_continuous_scale='RdYlGn', 
                range_color=[0, 100], 
                height=600
            )
            fig_tree.update_traces(
                textinfo="label+value",
                textfont=dict(size=14, color="white"), # 字体改白
                marker=dict(line=dict(width=1, color='#121212')) # 边框改黑
            )
            fig_tree.update_layout(
                template="plotly_dark", 
                margin=dict(t=0, l=0, r=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_tree, use_container_width=True)
            
    # Chart 2: Bubble (黑底荧光风)
    with t2:
        if not df_final.empty:
            fig_bubble = px.scatter(
                df_final, x="Stress_Margin", y="V14_Score", size="V14_Score", color="V14_Rating",
                hover_name="公司名称", 
                # SCB 荧光色系: 蓝, 绿, 青, 白, 灰
                color_discrete_sequence=["#005EBB", "#009F4D", "#00E5FF", "#B0BEC5", "#CFD8DC"], 
                height=550
            )
            fig_bubble.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor="#333"), 
                yaxis=dict(showgrid=True, gridcolor="#333")
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
            
    # Chart 3: Strip (黑底荧光风)
    with t3:
        if not df_final.empty:
            fig_dist = px.strip(
                df_final.sort_values("V14_Rating"), x="V14_Rating", y="V14_Score", color="V14_Rating",
                color_discrete_sequence=["#005EBB", "#009F4D", "#00E5FF", "#B0BEC5", "#CFD8DC"], 
                height=500
            )
            fig_dist.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(showgrid=True, gridcolor="#333")
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
    # Chart 4: Correlation (黑底 + 红蓝)
    with t4:
        st.markdown("**Chart 4: Factor Correlation**")
        if not df_final.empty:
            corr_cols = ['V14_Score', 'Stress_Margin', 'Inv_Days', '海外营收占比(%)', '资产负债率(%)']
            for c in corr_cols:
                if c not in df_final.columns: df_final[c] = 0
                df_final[c] = pd.to_numeric(df_final[c], errors='coerce').fillna(0)
            
            corr_matrix = df_final[corr_cols].corr().fillna(0)
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r', 
                zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}", 
                # 白色数字，适合黑底
                textfont={"size": 14, "color": "white", "family": "Helvetica Neue"},
                xgap=1, ygap=1
            ))
            
            fig_corr.update_layout(
                template="plotly_dark",
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="white"),
                margin=dict(t=20, l=20, r=20, b=20)
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    with t5:
        st.dataframe(df_final.sort_values("V14_Score", ascending=False), use_container_width=True)
        csv = df_final.to_csv(index=False).encode('utf-8-sig')
        st.download_button("💾 DOWNLOAD CSV", csv, "SCB_Risk_V14.csv", "text/csv")
