import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import uuid
from datetime import datetime
from fpdf import FPDF
import io
import graphviz # 新增：用于绘制架构图

# ==========================================
# 0. 系统配置 (V26.1 最终交付版)
# ==========================================
st.set_page_config(page_title="Global Credit Lens V26.1", layout="wide", page_icon="🏦")

# CSS 样式: 黑金/投行风
st.markdown("""
    <style>
    .stApp { background-color: #000000 !important; color: #E0E0E0; font-family: 'Microsoft YaHei', sans-serif; }
    [data-testid="stSidebar"] { background-color: #121212 !important; border-right: 1px solid #333; }
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 700 !important; letter-spacing: 1px; }
    .stMetric { background-color: #1A1A1A; border: 1px solid #333; border-left: 4px solid #0056D2; padding: 15px; border-radius: 5px; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1A1A1A; border-radius: 4px 4px 0 0; color: #888; }
    .stTabs [aria-selected="true"] { background-color: #0056D2 !important; color: white !important; }
    .stButton>button { background-color: #222; color: white; border: 1px solid #444; border-radius: 4px; }
    .stButton>button:hover { border-color: #0056D2; color: #0056D2; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 鉴权 & 数据加载
# ==========================================
def check_password():
    CORRECT_PASSWORD = "HR2026"
    if "password_correct" not in st.session_state:
        st.text_input("🔒 Access Key", type="password", key="password", on_change=lambda: st.session_state.update({"password_correct": st.session_state["password"] == CORRECT_PASSWORD}))
        return False
    return st.session_state["password_correct"]

if not check_password(): st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    if 'Ticker' not in df.columns: df['Ticker'] = "N/A"
    df['Ticker'] = df['Ticker'].astype(str).str.replace('.0', '', regex=False)
    return df

# ==========================================
# 2. 核心计算引擎 (Logit + Stress)
# ==========================================
class CreditEngine:
    @staticmethod
    def sigmoid(z): return 1 / (1 + np.exp(-z))

    @staticmethod
    def calculate(row, params, macro_status):
        try:
            # 基础指标
            base_gm = float(row.get('Gross Margin', 0))       
            debt_ratio = float(row.get('Debt Ratio', 50))     
            overseas = float(row.get('Overseas Ratio', 0))    
            inv = float(row.get('Inventory Days', 90))        
            cf = float(row.get('Cash Flow', 0))               
            cf_flag = 1 if cf > 0 else 0
        except: return pd.Series({'Score': 0, 'Rating': 'Error', 'PD_Prob': 1.0, 'Stressed_GM': 0})

        # 压力测试
        market_hit = params.get('margin_shock', 0) / 100.0
        tariff_hit = (overseas / 100.0) * params.get('tariff_shock', 0) * 100
        input_cost_hit = params.get('raw_material_shock', 0) * 0.2
        fx_hit = (overseas / 100.0) * params.get('fx_shock', 0) 

        final_gm = max(base_gm - market_hit - tariff_hit - input_cost_hit - fx_hit, -10.0)
        rate_hit = (debt_ratio / 100.0) * (params.get('rate_hike_bps', 0) / 100.0) * 5.0

        # Logit 公式
        logit_z = -0.5 + (-0.15 * final_gm) + (0.02 * inv) + (0.05 * debt_ratio) + (-1.2 * cf_flag) + rate_hit
        pd_val = CreditEngine.sigmoid(logit_z)
        score = 100 * (1 - pd_val)
        
        if score >= 85: rating = "AAA"
        elif score >= 70: rating = "AA"
        elif score >= 55: rating = "BBB"
        elif score >= 40: rating = "BB"
        else: rating = "CCC"
        
        return pd.Series({'Stressed_GM': final_gm, 'PD_Prob': pd_val, 'Score': score, 'Rating': rating})

# ==========================================
# 3. 巴塞尔资本引擎 (Basel III RWA)
# ==========================================
class BaselEngine:
    """
    巴塞尔协议 III 标准法资本计算器
    """
    def __init__(self):
        # 风险权重映射表 (Standardized Approach)
        self.rw_map = {
            'AAA': 0.20, 'AA': 0.20,
            'A': 0.50, 'BBB': 1.00,
            'BB': 1.00, 'B': 1.50,
            'CCC': 1.50, 'CC': 2.50, 'C': 2.50
        }
        self.capital_ratio = 0.08 # 最低资本充足率 8%

    def calculate_rwa(self, exposure, rating):
        # 1. 查找风险权重 (Risk Weight)
        rw = self.rw_map.get(rating, 1.50) # 默认高风险
        # 2. 计算 RWA
        rwa = exposure * rw
        # 3. 计算资本占用 (Capital Charge)
        charge = rwa * self.capital_ratio
        return rw, rwa, charge

# ==========================================
# 4. IV 计算引擎
# ==========================================
class IV_Engine:
    @staticmethod
    def calculate_iv(df, target_col='Is_Bad', feature_cols=[]):
        iv_list = []
        for col in feature_cols:
            try:
                temp_df = df[[col, target_col]].copy()
                temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0)
                try: temp_df['bucket'] = pd.qcut(temp_df[col], q=4, duplicates='drop')
                except: temp_df['bucket'] = pd.cut(temp_df[col], bins=4)
                
                grouped = temp_df.groupby('bucket', observed=False)[target_col].agg(['count', 'sum'])
                grouped['bad'] = grouped['sum']
                grouped['good'] = grouped['count'] - grouped['sum']
                total_bad = grouped['bad'].sum() + 1e-5
                total_good = grouped['good'].sum() + 1e-5
                
                grouped['dist_bad'] = (grouped['bad'] + 1e-5) / total_bad
                grouped['dist_good'] = (grouped['good'] + 1e-5) / total_good
                grouped['woe'] = np.log(grouped['dist_good'] / grouped['dist_bad'])
                grouped['iv'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['woe']
                iv_list.append({'Feature': col, 'IV': grouped['iv'].sum()})
            except: continue
        return pd.DataFrame(iv_list).sort_values(by='IV', ascending=False)

# ==========================================
# 5. 主程序
# ==========================================
def main():
    st.sidebar.title("🎛️ 压力测试实验室")
    
    # A. 数据接入
    st.sidebar.subheader("1. 数据接入")
    uploaded_file = st.sidebar.file_uploader("上传 Excel", type=['xlsx'])
    if uploaded_file: df_raw = load_data(uploaded_file)
    else:
        df_raw = pd.DataFrame([
            {'Ticker': '600438', 'Company': '通威股份', 'Gross Margin': 28.5, 'Overseas Ratio': 25.0, 'Inventory Days': 85, 'Debt Ratio': 55.0, 'Cash Flow': 1},
            {'Ticker': '300750', 'Company': '宁德时代', 'Gross Margin': 22.0, 'Overseas Ratio': 35.0, 'Inventory Days': 70, 'Debt Ratio': 45.0, 'Cash Flow': 1},
            {'Ticker': '601012', 'Company': '隆基绿能', 'Gross Margin': 18.0, 'Overseas Ratio': 45.0, 'Inventory Days': 95, 'Debt Ratio': 50.0, 'Cash Flow': 1},
            {'Ticker': '688599', 'Company': '天合光能', 'Gross Margin': 16.0, 'Overseas Ratio': 60.0, 'Inventory Days': 80, 'Debt Ratio': 65.0, 'Cash Flow': 1},
            {'Ticker': '002459', 'Company': '晶澳科技', 'Gross Margin': 15.5, 'Overseas Ratio': 55.0, 'Inventory Days': 88, 'Debt Ratio': 60.0, 'Cash Flow': 0}
        ])

    # B. 压力参数
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. 宏观压力参数")
    params = {
        'margin_shock': st.sidebar.slider("1. 行业内卷 (bps)", 0, 1000, 300),
        'tariff_shock': st.sidebar.slider("2. 关税壁垒 (%)", 0.0, 1.0, 0.25),
        'rate_hike_bps': st.sidebar.slider("3. 美联储加息 (bps)", 0, 500, 100),
        'raw_material_shock': st.sidebar.slider("4. 原材料通胀 (%)", 0, 50, 10),
        'fx_shock': st.sidebar.slider("5. 汇率风险 (%)", 0, 20, 5)
    }

    # C. 计算 (双重计算：Base vs Stressed)
    try:
        # 1. 压力环境 (Stressed)
        res_stressed = df_raw.apply(lambda r: CreditEngine.calculate(r, params, "Stressed"), axis=1)
        df_final = pd.concat([df_raw, res_stressed], axis=1)
        
        # 2. 基准环境 (Base Case - 所有参数归零)
        base_params = {k:0 for k in params}
        res_base = df_raw.apply(lambda r: CreditEngine.calculate(r, base_params, "Base"), axis=1)
        df_final['Base_Rating'] = res_base['Rating'] # 保存基准评级用于对比
        
        df_final['Search_Label'] = df_final['Ticker'] + " | " + df_final['Company']
    except: return

    # D. 界面展示
    st.title("GLOBAL CREDIT LENS | V26.1")
    st.caption(f"架构: Logit + Stress Test + Basel III RWA + Architecture View")
    
    # 检索
    c_search, _ = st.columns([1, 2])
    with c_search:
        selected_label = st.selectbox("🔍 穿透式检索", df_final['Search_Label'].tolist())
    
    selected_ticker = selected_label.split(" | ")[0]
    row = df_final[df_final['Ticker'] == selected_ticker].iloc[0]

    # --- 资本画像卡片 ---
    col1, col2 = st.columns([1, 2])
    with col1:
        # 假设贷款敞口
        exposure = st.number_input("💰 假设贷款敞口 (Exposure, USD)", value=10_000_000, step=1_000_000)
        
        # 调用 Basel 引擎
        basel = BaselEngine()
        rw_base, rwa_base, cap_base = basel.calculate_rwa(exposure, row['Base_Rating'])
        rw_stress, rwa_stress, cap_stress = basel.calculate_rwa(exposure, row['Rating'])
        cap_delta = cap_stress - cap_base

        rating_color = '#28A745' if row['Score'] >= 70 else '#DC3545'
        
        st.markdown(f"""
            <div style="background-color:#1A1A1A; padding:20px; border-radius:8px; border:1px solid #333;">
                <h4 style="color:#888; margin:0;">{row['Ticker']}</h4>
                <h2 style="color:white; margin:5px 0;">{row['Company']}</h2>
                <div style="margin-top:15px; padding:10px; background-color:{rating_color}20; border-left:4px solid {rating_color};">
                    <h1 style="color:{rating_color}; margin:0; font-size:48px;">{row['Rating']}</h1>
                </div>
                <p style="color:#AAA; margin-top:5px;">Base Rating: <b>{row['Base_Rating']}</b></p>
                <hr style="border-color:#333;">
                <p style="color:#EEE; font-size:24px; margin:5px 0;">RW: {rw_base:.0%} ➔ <span style="color:#FF4B4B">{rw_stress:.0%}</span></p>
                <p style="color:#AAA; font-size:14px;">Capital Charge Delta: <b style="color:#FF4B4B">+${cap_delta:,.0f}</b></p>
            </div>
        """, unsafe_allow_html=True)

        # PDF 导出逻辑
        if st.button(f"📄 导出 {row['Ticker']} 深度研报"):
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 20)
                pdf.cell(0, 15, f"CREDIT & CAPITAL MEMO: {row['Ticker']}", 0, 1)
                pdf.line(10, 25, 200, 25)
                pdf.ln(5)
                
                pdf.set_font("Arial", "", 12)
                pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
                pdf.cell(0, 10, f"Stressed Rating: {row['Rating']} (Base: {row['Base_Rating']})", 0, 1)
                
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 15, "CAPITAL IMPACT (BASEL III):", 0, 1)
                pdf.set_font("Arial", "", 12)
                pdf.cell(0, 8, f"Exposure: ${exposure:,.0f}", 0, 1)
                pdf.cell(0, 8, f"Risk Weight Change: {rw_base:.0%} -> {rw_stress:.0%}", 0, 1)
                pdf.cell(0, 8, f"Capital Charge (Base): ${cap_base:,.0f}", 0, 1)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, f"Capital Charge (Stressed): ${cap_stress:,.0f}", 0, 1)
                pdf.set_text_color(255, 0, 0)
                pdf.cell(0, 8, f"Additional Capital Required: +${cap_delta:,.0f}", 0, 1)
                pdf.set_text_color(0, 0, 0)
                
                pdf.ln(10)
                pdf.set_font("Arial", "I", 8)
                pdf.cell(0, 10, "Generated by Global Credit Lens V26.1", 0, 1)
                
                pdf_bytes = bytes(pdf.output())
                st.download_button("📥 下载 PDF", pdf_bytes, f"Capital_Memo_{row['Ticker']}.pdf", "application/pdf")
            except: pass

    with col2:
        # RWA 瀑布图
        fig = go.Figure(go.Waterfall(
            name = "20", orientation = "v",
            measure = ["relative", "relative", "total"],
            x = ["Base Capital", "Stress Impact", "Final Capital"],
            textposition = "outside",
            text = [f"${cap_base/1000:.0f}k", f"+${cap_delta/1000:.0f}k", f"${cap_stress/1000:.0f}k"],
            y = [cap_base, cap_delta, cap_stress],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
            increasing = {"marker":{"color":"#FF4B4B"}},
            decreasing = {"marker":{"color":"#28A745"}},
            totals = {"marker":{"color":"#333333"}}
        ))
        fig.update_layout(title = "压力测试下的资本损耗 (Capital Erosion)", template="plotly_dark", height=400, showlegend = False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    # 宏观 Tab 页
    st.markdown("---")
    st.subheader("📊 深度量化看板")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ 全景热力", "🛁 竞争气泡", "🎻 评级分布", "🔗 归因分析", "🧠 因子筛选(IV)"])

    with tab1:
        if not df_final.empty:
            st.plotly_chart(px.treemap(df_final, path=[px.Constant("全市场"), 'Rating', 'Search_Label'], values='Score', color='Score', color_continuous_scale='RdYlGn', title="信用风险分布"), use_container_width=True)
    with tab2:
        if not df_final.empty:
            st.plotly_chart(px.scatter(df_final, x="Stressed_GM", y="Score", size="Debt Ratio", color="Rating", hover_name="Company", text="Company", title="盈利能力 vs 评分"), use_container_width=True)
    with tab3:
        if not df_final.empty:
            st.plotly_chart(px.strip(df_final, x="Rating", y="Score", color="Rating", title="评级分布"), use_container_width=True)
    with tab4:
        if not df_final.empty:
            st.plotly_chart(px.imshow(df_final[['Score', 'Gross Margin', 'Overseas Ratio', 'Inventory Days', 'Debt Ratio']].corr(), text_auto=True, color_continuous_scale='RdBu_r', title="因子相关性"), use_container_width=True)
    with tab5:
        st.markdown("### 🧬 特征重要性分析 (IV)")
        if not df_final.empty:
            target_col = 'Manual_Bad_Label' if 'Manual_Bad_Label' in df_final.columns else 'Is_Bad'
            if target_col == 'Is_Bad': df_final['Is_Bad'] = df_final['PD_Prob'].apply(lambda x: 1 if x > 0.30 else 0)
            
            iv_result = IV_Engine.calculate_iv(df_final, target_col=target_col, feature_cols=['Gross Margin', 'Debt Ratio', 'Overseas Ratio', 'Inventory Days', 'Cash Flow'])
            iv_result['Color'] = iv_result['IV'].apply(lambda x: '#FFD700' if x > 0.3 else ('#00E5FF' if x > 0.1 else '#555555'))
            
            c_iv1, c_iv2 = st.columns([2, 1])
            with c_iv1:
                st.plotly_chart(px.bar(iv_result, x='IV', y='Feature', orientation='h', title="关键风险因子预测力 (IV)", text_auto='.3f', color='Feature', color_discrete_map={row['Feature']: row['Color'] for _, row in iv_result.iterrows()}).update_layout(template="plotly_dark", showlegend=False), use_container_width=True)
            with c_iv2:
                st.info("💡 **IV 标准:**\n- **>0.3**: 强因子 (金)\n- **0.1-0.3**: 有效 (蓝)\n- **<0.02**: 噪音")
                st.dataframe(iv_result[['Feature', 'IV']], use_container_width=True)

    # ==========================================
    # 6. 架构图绘制 (Graphviz Integration)
    # ==========================================
    st.markdown("---")
    st.subheader("🏗️ System Architecture (V26.1)")
    with st.expander("点击展开系统架构图 (Architect View)", expanded=False):
        try:
            arch = graphviz.Digraph()
            arch.attr(rankdir='TB', bgcolor='transparent')
            arch.attr('node', shape='box', style='filled', fontname='Arial', color='white', fontcolor='black')
            
            # 1. User Layer (绿色)
            with arch.subgraph(name='cluster_0') as c:
                c.attr(label='Presentation Layer', color='white', fontcolor='white')
                c.node('UI', 'Streamlit Dashboard\n(Web Interface)', fillcolor='#28A745', fontcolor='white')
                c.node('Report', 'Smart Report Engine\n(FPDF Generator)', fillcolor='#28A745', fontcolor='white')

            # 2. Core Layer (蓝色)
            with arch.subgraph(name='cluster_1') as c:
                c.attr(label='Core Computing Layer', color='white', fontcolor='white')
                c.node('Stress', 'Stress Engine\n(5-Factor Shock)', fillcolor='#0056D2', fontcolor='white')
                c.node('Logit', 'Scoring Engine\n(Logistic Regression)', fillcolor='#0056D2', fontcolor='white')
                c.node('Basel', 'Capital Engine\n(Basel III RWA)', fillcolor='#0056D2', fontcolor='white')
                c.node('Validation', 'Validation Engine\n(WOE / IV Analysis)', fillcolor='#0056D2', fontcolor='white')

            # 3. Data Layer (金色)
            with arch.subgraph(name='cluster_2') as c:
                c.attr(label='Data Ingestion', color='white', fontcolor='white')
                c.node('Excel', 'Excel Upload', fillcolor='#FFD700')
                c.node('SQL', 'SQL Adapter\n(Enterprise Ready)', fillcolor='#FFD700')
                c.node('Clean', 'Preprocessing', fillcolor='#FFD700')

            arch.edge('Excel', 'Clean')
            arch.edge('SQL', 'Clean')
            arch.edge('Clean', 'Validation')
            arch.edge('Clean', 'Stress')
            arch.edge('Validation', 'Logit')
            arch.edge('Stress', 'Logit')
            arch.edge('Logit', 'Basel')
            arch.edge('Basel', 'UI')
            arch.edge('Logit', 'UI')
            arch.edge('UI', 'Report')

            st.graphviz_chart(arch)
        except Exception as e:
            st.error("⚠️ 架构图渲染失败。请确保已安装 Graphviz (pip install graphviz)，并且系统环境变量已配置。")

if __name__ == "__main__":
    main()
