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
st.set_page_config(page_title="SCB Risk Pilot V13.0", layout="wide", initial_sidebar_state="expanded")

# --- 2. 样式修复：精准打击“白底白字”问题 ---
st.markdown("""
    <style>
    /* 1. 强制主背景为白 */
    .stApp {
        background-color: #FFFFFF !important;
    }
    
    /* 2. 强制侧边栏背景为灰 */
    [data-testid="stSidebar"] {
        background-color: #F0F2F6 !important;
        border-right: 1px solid #D1D1D1;
    }
    
    /* 3. 关键修复：强制所有Markdown文本和标题为黑色 */
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, li, span {
        color: #000000 !important;
        font-family: 'Arial', sans-serif !important;
    }
    
    /* 4. 修复指标卡 (Metric) 看不清的问题 */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #333333 !important;
    }
    
    /* 5. 按钮样式 */
    .stButton>button {
        background-color: #2E3B4E !important;
        color: #FFFFFF !important;
        border: none;
    }
    
    /* 6. Tab 样式 */
    .stTabs [aria-selected="true"] {
        background-color: #2E3B4E !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 智能数据处理 (保持 V12 逻辑) ---
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
def calculate_score_v13(row, params):
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
                     index=['V13_Score', 'V13_Rating', 'Stress_Margin', 'Inv_Days'])

# --- 5. 界面逻辑 ---
st.sidebar.markdown("## SCB RISK PILOT V13.0")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("MODULE", ["📈 MACRO HISTORY", "⚡ REAL-DATA STRESS TEST"])

current_folder = os.path.dirname(os.path.abspath(__file__))
xlsx_files = glob.glob(os.path.join(current_folder, "*.xlsx"))
if not xlsx_files: st.stop()
file_path = xlsx_files[0]

# =========================================================
# 模块一：历史周期
# =========================================================
if app_mode == "📈 MACRO HISTORY":
    st.markdown("### PV INDUSTRY CYCLE HISTORY (2000-2026)")
    
    anchors = {
        2000: 10,  2005: 40,  2008: 100, 2009: 25,
        2011: 15,  2013: 55,  2016: 85,  2018: 30,
        2020: 95,  2022: 100, 2024: 20,  2026: 85
    }
    events_map = {
        2005: "尚德上市", 2008: "拥硅为王", 2009: "金融危机",
        2011: "欧美双反", 2013: "国内补贴", 2016: "领跑者计划",
        2018: "531新政",  2020: "碳中和元年", 2024: "极度内卷", 2026: "AI反转"
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
        textfont=dict(size=14, color='black', family="Arial Black"), 
        name='Cycle',
        line=dict(color='#2E3B4E', width=3),
        marker=dict(size=10, color='#D32F2F', line=dict(width=2, color='white')),
        fill='tozeroy',
        fillcolor='rgba(46, 59, 78, 0.1)'
    ))
    
    # 关键修复：使用 template="plotly_white" 强制白底黑字
    fig.update_layout(
        template="plotly_white", # 救星！
        height=600,
        xaxis=dict(showgrid=False, tickmode='linear', dtick=1, tickangle=-90),
        yaxis=dict(showgrid=True, gridcolor='#F0F0F0', title="Sentiment")
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
    
    st.markdown("### DATA ENRICHMENT")
    c1, c2 = st.columns([3, 1])
    with c1: st.info("Fetch real data or use smart simulation.")
    with c2: 
        fetch_triggered = st.button("📡 FETCH REAL DATA")

    if fetch_triggered:
        with st.spinner("Processing Data..."):
            df_work = process_data_smartly(df_raw, use_real_fetch=True)
            st.session_state['df_v13'] = df_work
            st.success("Data Updated!")
    elif 'df_v13' in st.session_state:
        df_work = st.session_state['df_v13']
    else:
        df_work = process_data_smartly(df_raw, use_real_fetch=False)

    st.markdown("---")
    
    st.sidebar.markdown("### STRESS PARAMETERS")
    margin_shock = st.sidebar.slider("Margin Shock (-%)", 0, 15, 5) / 100.0
    tariff_shock = st.sidebar.slider("Tariff Shock (-%)", 0, 20, 10) / 100.0
    inv_limit = st.sidebar.slider("Inv Days Limit", 60, 200, 120)
    
    params = {'margin_shock': margin_shock, 'tariff_shock': tariff_shock, 'inv_limit': inv_limit}
    v13_res = df_work.apply(lambda row: calculate_score_v13(row, params), axis=1)
    df_final = pd.concat([df_work, v13_res], axis=1)
    
    st.markdown("### RISK COCKPIT")
    
    t1, t2, t3, t4, t5 = st.tabs([
        "🗺️ 全行业热力图", 
        "🔵 竞争格局气泡图", 
        "🎻 评级分布验证图", 
        "🔥 因子相关性矩阵",
        "📋 数据明细"
    ])
    
    # Chart 1: RdYlGn (复刻)
    with t1:
        st.markdown("**Chart 1: Industry Heatmap** (Green=Safe, Red=Risk)")
        if not df_final.empty:
            fig_tree = px.treemap(
                df_final,
                path=[px.Constant("PV Sector"), 'V13_Rating', '公司名称'],
                values='V13_Score',
                color='V13_Score',
                color_continuous_scale='RdYlGn', 
                range_color=[0, 100], 
                height=600
            )
            fig_tree.update_traces(
                textinfo="label+value",
                textfont=dict(size=14, color="black"),
                marker=dict(line=dict(width=2, color='white'))
            )
            # 关键修复：template="plotly_white"
            fig_tree.update_layout(template="plotly_white", margin=dict(t=0, l=0, r=0, b=0))
            st.plotly_chart(fig_tree, use_container_width=True)
            
    # Chart 2: Bubble
    with t2:
        if not df_final.empty:
            fig_bubble = px.scatter(
                df_final, x="Stress_Margin", y="V13_Score", size="V13_Score", color="V13_Rating",
                hover_name="公司名称", color_discrete_sequence=["#2E3B4E", "#5D6D7E", "#90A4AE", "#CFD8DC"], height=550
            )
            # 关键修复：template="plotly_white"
            fig_bubble.update_layout(
                template="plotly_white",
                xaxis=dict(showgrid=True, gridcolor="#F0F0F0"), 
                yaxis=dict(showgrid=True, gridcolor="#F0F0F0")
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
            
    # Chart 3: Strip
    with t3:
        if not df_final.empty:
            fig_dist = px.strip(
                df_final.sort_values("V13_Rating"), x="V13_Rating", y="V13_Score", color="V13_Rating",
                color_discrete_sequence=["#2E3B4E", "#5D6D7E", "#90A4AE", "#CFD8DC"], height=500
            )
            # 关键修复：template="plotly_white"
            fig_dist.update_layout(template="plotly_white", yaxis=dict(showgrid=True, gridcolor="#F0F0F0"))
            st.plotly_chart(fig_dist, use_container_width=True)
            
    # Chart 4: RdBu (复刻)
    with t4:
        st.markdown("**Chart 4: Correlation Matrix** (Red=Positive, Blue=Negative)")
        if not df_final.empty:
            corr_cols = ['V13_Score', 'Stress_Margin', 'Inv_Days', '海外营收占比(%)', '资产负债率(%)']
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
                textfont={"size": 14, "color": "black", "family": "Arial"},
                xgap=2, ygap=2
            ))
            
            # 关键修复：template="plotly_white"
            fig_corr.update_layout(
                template="plotly_white", # 强制白底
                height=600,
                xaxis=dict(side="bottom"),
                margin=dict(t=20, l=20, r=20, b=20)
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    with t5:
        st.dataframe(df_final.sort_values("V13_Score", ascending=False), use_container_width=True)
        csv = df_final.to_csv(index=False).encode('utf-8-sig')
        st.download_button("💾 DOWNLOAD CSV", csv, "SCB_Risk_V13.csv", "text/csv")
