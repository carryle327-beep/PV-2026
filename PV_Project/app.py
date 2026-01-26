import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import glob
import time
import akshare as ak

# --- 1. 页面配置 ---
st.set_page_config(page_title="SCB Risk Pilot V8.0", layout="wide", initial_sidebar_state="expanded")

# --- 2. 极致机构灰 CSS (保持高级冷淡风，但允许图表彩色) ---
st.markdown("""
    <style>
    .stApp { background-color: #F5F7F9 !important; }
    html, body, p, h1, h2, h3, h4, h5, h6, span, div, label {
        color: #000000 !important; font-family: 'Helvetica Neue', Arial, sans-serif !important;
    }
    section[data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #D1D1D1; }
    
    /* 控件去色，保持专业 */
    div[data-baseweb="slider"] div[class*="css-"] { background-color: #2E3B4E !important; }
    div[role="slider"] { background-color: #2E3B4E !important; border-color: #2E3B4E !important; }
    
    /* 按钮：深岩灰 */
    .stButton>button {
        background-color: #2E3B4E !important; color: #FFFFFF !important; border-radius: 2px;
        padding: 8px 16px; font-weight: 600; text-transform: uppercase;
    }
    .stButton>button:hover { background-color: #1C2430 !important; }
    
    /* 指标卡 */
    div[data-testid="stMetric"] { background-color: #FFFFFF !important; border: 1px solid #D1D1D1; padding: 15px; }
    div[data-testid="stMetricValue"] { font-size: 26px !important; }
    
    /* Tab 样式 */
    .stTabs [aria-selected="true"] {
        background-color: #2E3B4E !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 真实数据获取引擎 ---
@st.cache_data(ttl=3600)
def fetch_real_company_data(stock_code):
    code = str(stock_code).split(".")[0].zfill(6)
    data = {'real_inventory_days': np.nan, 'real_overseas_ratio': 0.0, 'real_gross_margin': np.nan}
    try:
        # 1. 财务指标
        df_fin = ak.stock_financial_analysis_indicator(symbol=code)
        if not df_fin.empty:
            latest = df_fin.iloc[0]
            if '存货周转天数(天)' in latest: data['real_inventory_days'] = float(latest['存货周转天数(天)'])
            if '销售毛利率(%)' in latest: data['real_gross_margin'] = float(latest['销售毛利率(%)'])
        # 2. 海外占比
        df_biz = ak.stock_zygc_em(symbol=code)
        if not df_biz.empty:
            mask = df_biz.astype(str).apply(lambda x: x.str.contains('外').any(), axis=1)
            for idx, row in df_biz[mask].iterrows():
                for item in row:
                    if isinstance(item, str) and "%" in item:
                        try:
                            val = float(item.strip('%'))
                            if val > data['real_overseas_ratio']: data['real_overseas_ratio'] = val
                        except: continue
        return data
    except: return data

def batch_fetch_data(df):
    if '股票代码' not in df.columns: return df
    progress_bar = st.progress(0)
    status_text = st.empty()
    real_data_list = []
    for i, row in df.iterrows():
        status_text.text(f"Fetching: {row['公司名称']}...")
        real_data_list.append(fetch_real_company_data(row['股票代码']))
        progress_bar.progress((i + 1) / len(df))
        time.sleep(0.05)
    progress_bar.empty()
    status_text.empty()
    
    df_real = pd.DataFrame(real_data_list)
    df_final = pd.concat([df.reset_index(drop=True), df_real], axis=1)
    
    # 数据补全
    if '存货周转天数' not in df_final.columns: df_final['存货周转天数'] = 90
    df_final['存货周转天数'] = df_final['real_inventory_days'].fillna(df_final['存货周转天数'])
    df_final['海外营收占比(%)'] = df_final['real_overseas_ratio'].fillna(0)
    df_final['最新毛利率'] = df_final['real_gross_margin'].fillna(df_final['技术壁垒(毛利率%)'])
    return df_final

# --- 4. 评分引擎 ---
def calculate_score_v8(row, params):
    score = 0
    reasons = []
    base_margin = row.get('最新毛利率', row['技术壁垒(毛利率%)'])
    
    # 压力测试
    stress_margin = base_margin - (params['margin_shock'] * 100)
    if row.get('海外营收占比(%)', 0) > 50:
        stress_margin -= (params['tariff_shock'] * 100)
        reasons.append("Tariff Hit")
        
    # 赛道评分
    is_equipment = any(x in str(row['公司名称']) for x in ['设备', '激光', '机', '微导', '捷佳', '奥特维'])
    if is_equipment:
        if stress_margin >= 30: score += 40
        elif stress_margin >= 20: score += 20
    else:
        if stress_margin >= 15: score += 30
        elif stress_margin >= 10: score += 15
        
    # 库存与现金流
    inv = row.get('存货周转天数', 90)
    if inv > params['inv_limit']: 
        score -= 15
        reasons.append(f"High Inv ({inv:.0f}d)")
    else: score += 10
    
    if row.get('每股经营现金流(元)', 0) < 0: 
        score -= 20
        reasons.append("CF Neg")
    else: score += 20
    
    # 第二曲线
    if row.get('第二曲线(储能)', False): score += 10
    
    final_score = min(100, max(0, score))
    
    if final_score >= 80: rating = "A (Priority)"
    elif final_score >= 60: rating = "B (Watch)"
    elif final_score >= 40: rating = "C (Prudent)"
    else: rating = "D (Exit)"
    
    return pd.Series([final_score, rating, stress_margin, inv, ", ".join(reasons)], 
                     index=['V8_Score', 'V8_Rating', 'Stress_Margin', 'Inv_Days', 'Risks'])

# --- 5. 界面逻辑 ---
st.sidebar.markdown("## SCB RISK PILOT V8.0")
st.sidebar.caption("HTML STYLE REPLICATION")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("MODULE", ["📈 MACRO HISTORY", "⚡ REAL-DATA STRESS TEST"])

current_folder = os.path.dirname(os.path.abspath(__file__))
xlsx_files = glob.glob(os.path.join(current_folder, "*.xlsx"))
if not xlsx_files: st.stop()
file_path = xlsx_files[0]

# =========================================================
# 模块一：历史周期 (大事件直接显示版)
# =========================================================
if app_mode == "📈 MACRO HISTORY":
    st.markdown("### PV INDUSTRY CYCLE HISTORY (2000-2026)")
    
    # 1. 构造高波动数据
    anchors = {
        2000: 10,  2005: 40,  2008: 100, 2009: 25,
        2011: 15,  2013: 55,  2016: 85,  2018: 30,
        2020: 95,  2022: 100, 2024: 20,  2026: 85
    }
    # 2. 构造大事件标签 (直接显示在图上)
    events_map = {
        2005: "尚德上市", 2008: "拥硅为王", 2009: "金融危机",
        2011: "欧美双反", 2013: "国内补贴", 2016: "领跑者计划",
        2018: "531新政",  2020: "碳中和元年", 2024: "极度内卷", 2026: "AI反转"
    }
    
    full_years = list(range(2000, 2027))
    s_val = pd.Series(anchors).reindex(full_years).interpolate(method='linear')
    s_event = pd.Series(events_map).reindex(full_years).fillna("") # 没有事件的填空字符串
    
    df_hist = pd.DataFrame({'year': full_years, 'val': s_val.values, 'label': s_event.values})
    
    fig = go.Figure()
    
    # 绘制折线 + 标记 + 文字
    fig.add_trace(go.Scatter(
        x=df_hist['year'], 
        y=df_hist['val'], 
        mode='lines+markers+text', # 关键：加上 text 模式
        text=df_hist['label'],     # 关键：绑定文字列
        textposition="top center", # 文字显示在点上方
        textfont=dict(size=12, color='#000000', family="Arial Black"), # 黑色加粗字体
        name='Cycle Index',
        line=dict(color='#2E3B4E', width=3),
        marker=dict(size=8, color='#D32F2F', line=dict(width=2, color='white')), # 红色点强调事件
        fill='tozeroy',
        fillcolor='rgba(46, 59, 78, 0.1)'
    ))
    
    fig.update_layout(
        plot_bgcolor='white', 
        height=600,
        title="China PV Industry Cycle (Volatility & Key Events)",
        xaxis=dict(showgrid=False, tickmode='linear', dtick=1, tickangle=-90),
        yaxis=dict(showgrid=True, gridcolor='#EEE', title="Industry Sentiment Index")
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 模块二：实战风控
# =========================================================
elif app_mode == "⚡ REAL-DATA STRESS TEST":
    try:
        sheet_names = pd.ExcelFile(file_path).sheet_names
        selected_sheet = st.sidebar.selectbox("DATA SHEET", sheet_names)
    except: st.stop()
    
    @st.cache_data
    def load_raw(p, s): return pd.read_excel(p, sheet_name=s)
    df_raw = load_raw(file_path, selected_sheet)
    
    # 数据获取
    st.markdown("### 1. DATA ENRICHMENT")
    c1, c2 = st.columns([3, 1])
    with c1: st.info("Fetch real-time financial data to power the charts below.")
    with c2: 
        if st.button("📡 FETCH REAL DATA"):
            with st.spinner("Crawling Data..."):
                df_proc = batch_fetch_data(df_raw)
                st.session_state['df_real'] = df_proc
                st.success("Data Fetched!")
                
    if 'df_real' in st.session_state:
        df_work = st.session_state['df_real']
    else:
        df_work = df_raw.copy()
        if '存货周转天数' not in df_work.columns: df_work['存货周转天数'] = 90
        df_work['海外营收占比(%)'] = 0
        df_work['最新毛利率'] = df_work['技术壁垒(毛利率%)']

    st.markdown("---")
    
    # 压力参数
    st.sidebar.markdown("### STRESS PARAMETERS")
    margin_shock = st.sidebar.slider("Margin Shock (-%)", 0, 15, 5) / 100.0
    tariff_shock = st.sidebar.slider("Tariff Shock (-%)", 0, 20, 10) / 100.0
    inv_limit = st.sidebar.slider("Inv Days Limit", 60, 200, 120)
    
    # 计算
    params = {'margin_shock': margin_shock, 'tariff_shock': tariff_shock, 'inv_limit': inv_limit}
    v8_res = df_work.apply(lambda row: calculate_score_v8(row, params), axis=1)
    df_final = pd.concat([df_work, v8_res], axis=1)
    
    # 结果区
    st.markdown("### 2. RISK VISUALIZATION COCKPIT")
    
    t1, t2, t3, t4, t5 = st.tabs([
        " 全行业信贷热力图", 
        " 竞争格局气泡图", 
        " 评级分布验证图", 
        " 因子相关性矩阵",
        " 数据明细"
    ])
    
    # Chart 1: 完美复刻上传文件 (RdYlGn)
    with t1:
        st.markdown("**Chart 1: Industry Credit Heatmap** (Green=Safe, Red=Risk)")
        if not df_final.empty:
            fig_tree = px.treemap(
                df_final,
                path=[px.Constant("PV Sector"), 'V8_Rating', '公司名称'],
                values='V8_Score',
                color='V8_Score',
                # 关键修改：强制使用红黄绿，复刻您上传的 HTML 风格
                color_continuous_scale='RdYlGn', 
                range_color=[0, 100], # 固定范围，保证颜色准
                hover_data=['Stress_Margin', 'Inv_Days'],
                height=550
            )
            fig_tree.update_layout(margin=dict(t=20, l=10, r=10, b=10))
            st.plotly_chart(fig_tree, use_container_width=True)
            
    # Chart 2: 保持 SCB 风格
    with t2:
        st.markdown("**Chart 2: Competition Landscape**")
        if not df_final.empty:
            fig_bubble = px.scatter(
                df_final,
                x="Stress_Margin",
                y="V8_Score",
                size="V8_Score",
                color="V8_Rating",
                hover_name="公司名称",
                # 这里保持冷色系以区分
                color_discrete_sequence=["#2E3B4E", "#5D6D7E", "#90A4AE", "#B0BEC5", "#CFD8DC"],
                height=550
            )
            fig_bubble.add_vline(x=15, line_dash="dot", line_color="#333")
            fig_bubble.add_hline(y=60, line_dash="dot", line_color="#333")
            fig_bubble.update_layout(plot_bgcolor="white", xaxis=dict(showgrid=True, gridcolor="#EEE"), yaxis=dict(showgrid=True, gridcolor="#EEE"))
            st.plotly_chart(fig_bubble, use_container_width=True)
            
    # Chart 3: 保持 SCB 风格
    with t3:
        st.markdown("**Chart 3: Rating Distribution**")
        if not df_final.empty:
            fig_dist = px.strip(
                df_final.sort_values("V8_Rating"),
                x="V8_Rating",
                y="V8_Score",
                color="V8_Rating",
                color_discrete_sequence=["#2E3B4E", "#5D6D7E", "#90A4AE", "#B0BEC5", "#CFD8DC"],
                height=500
            )
            fig_dist.update_layout(plot_bgcolor="white", yaxis=dict(showgrid=True, gridcolor="#EEE"))
            st.plotly_chart(fig_dist, use_container_width=True)
            
    # Chart 4: 完美复刻上传文件 (RdBu)
    with t4:
        st.markdown("**Chart 4: Factor Correlation Matrix** (Red=Positive, Blue=Negative)")
        if not df_final.empty:
            corr_cols = ['V8_Score', 'Stress_Margin', 'Inv_Days', '海外营收占比(%)', '资产负债率(%)']
            valid_cols = [c for c in corr_cols if c in df_final.columns]
            corr_matrix = df_final[valid_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                # 关键修改：强制使用红蓝对抗色，复刻您上传的 HTML 风格
                color_continuous_scale="RdBu_r", 
                zmin=-1, zmax=1, # 固定范围
                height=500
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    with t5:
        st.dataframe(df_final.sort_values("V8_Score", ascending=False), use_container_width=True)
        csv = df_final.to_csv(index=False).encode('utf-8-sig')
        st.download_button("💾 DOWNLOAD FULL REPORT", csv, "SCB_Risk_V8.csv", "text/csv")
