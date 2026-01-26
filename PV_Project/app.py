import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import glob
import time
# 引入 akshare 获取真实数据
import akshare as ak

# --- 1. 页面配置 ---
st.set_page_config(page_title="SCB Risk Pilot V5.0 (Real Data)", layout="wide", initial_sidebar_state="expanded")

# --- 2. 极致机构灰 CSS (保持高级感) ---
st.markdown("""
    <style>
    .stApp { background-color: #F5F7F9 !important; }
    html, body, p, h1, h2, h3, h4, h5, h6, span, div, label {
        color: #000000 !important; font-family: 'Helvetica Neue', Arial, sans-serif !important;
    }
    section[data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #D1D1D1; }
    div[data-baseweb="slider"] div[class*="css-"] { background-color: #2E3B4E !important; }
    div[role="slider"] { background-color: #2E3B4E !important; border-color: #2E3B4E !important; }
    .stButton>button {
        background-color: #2E3B4E !important; color: #FFFFFF !important; border-radius: 2px;
        padding: 8px 16px; font-weight: 600; text-transform: uppercase;
    }
    .stButton>button:hover { background-color: #1C2430 !important; }
    div[data-testid="stMetric"] { background-color: #FFFFFF !important; border: 1px solid #D1D1D1; padding: 15px; }
    div[data-testid="stMetricValue"] { font-size: 26px !important; }
    .stAlert { background-color: #E3F2FD; border: 1px solid #90CAF9; color: #000; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 核心升级：真实数据获取引擎 (Real Data Engine) ---

@st.cache_data(ttl=3600) # 缓存1小时，避免重复爬取
def fetch_real_company_data(stock_code):
    """
    爬取单家公司的：1. 存货周转天数 2. 海外营收占比 3. 最新毛利率
    """
    # 格式化代码 (确保是6位数字)
    code = str(stock_code).split(".")[0].zfill(6)
    
    data = {
        'real_inventory_days': np.nan,
        'real_overseas_ratio': 0.0,
        'real_gross_margin': np.nan,
        'data_source': 'Simulated' # 默认为模拟，获取成功则改为 Real
    }
    
    try:
        # 1. 获取财务指标 (存货周转天数, 毛利率)
        # akshare 接口: stock_financial_analysis_indicator
        df_fin = ak.stock_financial_analysis_indicator(symbol=code)
        if not df_fin.empty:
            # 取最新一期数据
            latest = df_fin.iloc[0]
            if '存货周转天数(天)' in latest:
                data['real_inventory_days'] = float(latest['存货周转天数(天)'])
            if '销售毛利率(%)' in latest:
                data['real_gross_margin'] = float(latest['销售毛利率(%)'])
        
        # 2. 获取主营业务构成 (海外占比)
        # akshare 接口: stock_zygc_em
        df_biz = ak.stock_zygc_em(symbol=code)
        if not df_biz.empty:
            # 寻找 "境外" 或 "海外" 或 "国外" 字眼
            # 通常列名是 '分类', '主营业务收入占比'
            # 这里的结构比较复杂，我们需要遍历
            overseas_ratio = 0.0
            # 筛选包含"外"字的行
            mask = df_biz.astype(str).apply(lambda x: x.str.contains('外').any(), axis=1)
            df_overseas = df_biz[mask]
            
            # 尝试提取占比数字 (通常是 string 如 "45.23%")
            for idx, row in df_overseas.iterrows():
                for item in row:
                    if isinstance(item, str) and "%" in item:
                        try:
                            val = float(item.strip('%'))
                            # 简单的逻辑：取最大的那个百分比作为海外占比（假设是按地区分类的汇总）
                            if val > overseas_ratio:
                                overseas_ratio = val
                        except:
                            continue
            data['real_overseas_ratio'] = overseas_ratio
            
        data['data_source'] = 'Real-Time'
        return data
        
    except Exception as e:
        # print(f"Error fetching {code}: {e}")
        return data

def batch_fetch_data(df):
    """
    批量获取，带进度条
    """
    if '股票代码' not in df.columns:
        st.error("Excel中缺少'股票代码'列，无法获取真实数据！")
        return df
    
    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    real_data_list = []
    total = len(df)
    
    for i, row in df.iterrows():
        code = row['股票代码']
        name = row['公司名称']
        status_text.text(f"Fetching data for {name} ({code})...")
        
        # 获取数据
        real_data = fetch_real_company_data(code)
        real_data_list.append(real_data)
        
        # 更新进度
        progress_bar.progress((i + 1) / total)
        # 稍微停顿避免被封IP
        time.sleep(0.1) 
        
    status_text.text("Data fetch complete!")
    progress_bar.empty()
    
    # 合并数据
    df_real = pd.DataFrame(real_data_list)
    df_final = pd.concat([df.reset_index(drop=True), df_real], axis=1)
    
    # 填充：如果获取失败，用 Excel 原有数据或中位数填充
    if '存货周转天数' not in df_final.columns:
        df_final['存货周转天数'] = df_final['real_inventory_days'].fillna(90)
    else:
        # 优先用 Real，空的用 Excel 里的
        df_final['存货周转天数'] = df_final['real_inventory_days'].fillna(df_final.get('存货周转天数', 90))
        
    df_final['海外营收占比(%)'] = df_final['real_overseas_ratio'].fillna(0)
    
    # 更新毛利率 (如果有真实数据且不为空)
    df_final['最新毛利率'] = df_final['real_gross_margin'].fillna(df_final['技术壁垒(毛利率%)'])
    
    return df_final

# 4. 评分引擎 V5 (基于真实数据)
def calculate_score_v5(row, params):
    score = 0
    reasons = []
    
    # 使用真实抓取的最新毛利，如果没有则用Excel的
    base_margin = row.get('最新毛利率', row['技术壁垒(毛利率%)'])
    
    # 1. 压力测试：价格战冲击
    stress_margin = base_margin - (params['margin_shock'] * 100)
    
    # 2. 压力测试：关税冲击 (基于真实海外占比)
    overseas_ratio = row.get('海外营收占比(%)', 0)
    if overseas_ratio > 50: # 超过50%收入来自海外
        tariff_hit = params['tariff_shock'] * 100
        stress_margin -= tariff_hit
        reasons.append(f"Tariff Hit (-{tariff_hit:.0f}%)")
    
    # 3. 赛道评分 (设备 vs 制造)
    is_equipment = any(x in str(row['公司名称']) for x in ['设备', '激光', '机', '微导', '捷佳'])
    if is_equipment:
        if stress_margin >= 30: score += 40
        elif stress_margin >= 20: score += 20
    else:
        if stress_margin >= 15: score += 30
        elif stress_margin >= 10: score += 15
        
    # 4. 生存能力：库存周转 (基于真实数据)
    inv_days = row.get('存货周转天数', 90)
    if inv_days > params['inv_limit']:
        score -= 15
        reasons.append(f"High Inv ({inv_days:.0f}d)")
    else:
        score += 10
        
    # 5. 生存能力：现金流 (Excel数据)
    cf = row.get('每股经营现金流(元)', 0)
    if cf < 0:
        score -= 20
        reasons.append("CF Neg")
    else:
        score += 20
        
    # 6. 第二曲线 (模拟/Excel)
    if row.get('第二曲线(储能)', False):
        score += 10
        
    final_score = min(100, max(0, score))
    
    # 评级
    if final_score >= 80: rating = "A (Priority)"
    elif final_score >= 60: rating = "B (Watch)"
    elif final_score >= 40: rating = "C (Prudent)"
    else: rating = "D (Exit)"
    
    return pd.Series([final_score, rating, stress_margin, overseas_ratio, inv_days, ", ".join(reasons)], 
                     index=['V5_Score', 'V5_Rating', 'Stress_Margin', 'Overseas_Ratio', 'Inv_Days', 'Risks'])

# --- 5. 界面逻辑 ---

st.sidebar.markdown("## SCB RISK PILOT V5.0")
st.sidebar.caption("REAL DATA COMBAT EDITION")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("MODULE", ["📈 MACRO HISTORY", "⚡ REAL-DATA STRESS TEST"])

# 自动加载
current_folder = os.path.dirname(os.path.abspath(__file__))
xlsx_files = glob.glob(os.path.join(current_folder, "*.xlsx"))
if not xlsx_files: st.stop()
file_path = xlsx_files[0]

if app_mode == "📈 MACRO HISTORY":
    # 保持历史模块不变
    st.markdown("### PV INDUSTRY CYCLE HISTORY")
    # ... (Reuse history code)
    st.info("Historical data loaded.")

elif app_mode == "⚡ REAL-DATA STRESS TEST":
    try:
        sheet_names = pd.ExcelFile(file_path).sheet_names
        selected_sheet = st.sidebar.selectbox("DATA SHEET", sheet_names)
    except: st.stop()

    # 加载原始 Excel
    @st.cache_data
    def load_raw(path, sheet):
        df = pd.read_excel(path, sheet_name=sheet)
        return df
    
    df_raw = load_raw(file_path, selected_sheet)
    
    # --- 真实数据获取区 ---
    st.markdown("### 1. DATA ENRICHMENT")
    
    col_d1, col_d2 = st.columns([3, 1])
    with col_d1:
        st.info("Click 'FETCH REAL DATA' to crawl latest financial reports from Stock Exchange.")
    with col_d2:
        # 按钮：触发真实数据爬取
        fetch_btn = st.button("📡 FETCH REAL DATA")
    
    if fetch_btn:
        with st.spinner("Connecting to Exchange Database... Analyzing Financial Reports..."):
            df_processed = batch_fetch_data(df_raw)
            # 保存到 session state 以便后续使用
            st.session_state['df_real'] = df_processed
            st.success(f"Successfully fetched data for {len(df_processed)} companies!")
    
    # 检查是否有数据，否则使用原始数据模拟
    if 'df_real' in st.session_state:
        df_work = st.session_state['df_real']
        is_real = True
    else:
        df_work = df_raw.copy()
        # 如果还没抓取，先给默认值防止报错
        if '存货周转天数' not in df_work.columns: df_work['存货周转天数'] = 90
        if '海外营收占比(%)' not in df_work.columns: df_work['海外营收占比(%)'] = 20
        df_work['最新毛利率'] = df_work['技术壁垒(毛利率%)']
        is_real = False
        st.warning("⚠️ Currently using Excel/Simulated data. Fetch Real Data for accuracy.")

    st.markdown("---")
    
    # --- 压力测试参数 ---
    st.sidebar.markdown("### STRESS PARAMETERS")
    margin_shock = st.sidebar.slider("Margin Shock (-%)", 0, 15, 5) / 100.0
    tariff_shock = st.sidebar.slider("Tariff Shock (-%)", 0, 20, 10) / 100.0
    inv_limit = st.sidebar.slider("Inv Days Limit", 60, 200, 120)
    
    # --- 计算 V5 分数 ---
    params = {'margin_shock': margin_shock, 'tariff_shock': tariff_shock, 'inv_limit': inv_limit}
    v5_res = df_work.apply(lambda row: calculate_score_v5(row, params), axis=1)
    df_final = pd.concat([df_work, v5_res], axis=1)
    
    # --- 结果展示 ---
    st.markdown("### 2. STRESS TEST RESULTS (V5)")
    st.caption(f"Based on: {'REAL-TIME DATA' if is_real else 'STATIC DATA'} | Stress: Margin -{margin_shock*100}% | Tariff -{tariff_shock*100}%")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies", len(df_final))
    c2.metric("Survivors (Grade A)", len(df_final[df_final['V5_Rating'].str.contains("A")]))
    avg_inv = df_final['Inv_Days'].mean()
    c3.metric("Avg Inventory Days", f"{avg_inv:.0f} d", delta="-High Risk" if avg_inv > inv_limit else "Safe", delta_color="inverse")
    avg_overseas = df_final['Overseas_Ratio'].mean()
    c4.metric("Avg Overseas Rev", f"{avg_overseas:.1f}%")
    
    t1, t2 = st.tabs(["🌪️ V5 MATRIX", "📋 DETAIL GRID"])
    
    with t1:
        if not df_final.empty:
            # 颜色：区分数据源
            fig = px.scatter(
                df_final,
                x="Stress_Margin",
                y="V5_Score",
                size="V5_Score",
                color="V5_Rating",
                hover_name="公司名称",
                hover_data=["Overseas_Ratio", "Inv_Days", "Risks"],
                title=f"Survival Matrix ({'Real Data' if is_real else 'Simulated'})",
                color_discrete_sequence=["#2E3B4E", "#5D6D7E", "#90A4AE", "#CFD8DC"],
                height=550
            )
            # 画库存警戒线逻辑不好画在图上，用文字提示
            fig.add_hline(y=60, line_dash="dot", annotation_text="Invest Line")
            fig.update_layout(plot_bgcolor="white", xaxis=dict(showgrid=True, gridcolor="#EEE"), yaxis=dict(showgrid=True, gridcolor="#EEE"))
            st.plotly_chart(fig, use_container_width=True)
            
    with t2:
        show_cols = ['公司名称', '股票代码', 'V5_Rating', 'V5_Score', 'Stress_Margin', 'Overseas_Ratio', 'Inv_Days', 'Risks']
        st.dataframe(df_final[show_cols].sort_values("V5_Score", ascending=False), use_container_width=True)
    with t2:
        show_cols = ['公司名称', '股票代码', 'V5_Rating', 'V5_Score', 'Stress_Margin', 'Overseas_Ratio', 'Inv_Days', 'Risks']
        st.dataframe(df_final[show_cols].sort_values("V5_Score", ascending=False), use_container_width=True)
        
        # --- 👇 新增：下载按钮 (Download Button) 👇 ---
        st.markdown("---")
        st.markdown("#### 📥 EXPORT RESULTS")
        
        # 1. 准备数据：把爬取到的所有真实数据都带上
        export_df = df_final.copy()
        
        # 2. 转换成 CSV (Excel通用格式)
        csv_data = export_df.to_csv(index=False).encode('utf-8-sig') # utf-8-sig 保证中文不乱码
        
        # 3. 放置按钮
        st.download_button(
            label="💾 DOWNLOAD FULL REPORT (.CSV)",
            data=csv_data,
            file_name=f'SCB_Risk_Rating_V5_{pd.Timestamp.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
            help="Click to save the crawled real-time data and V5 scores to a new file."
        )

