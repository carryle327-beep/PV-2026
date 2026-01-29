import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import uuid
from datetime import datetime
from fpdf import FPDF
import io

# ==========================================
# 0. 系统配置 (V25.1 智能报告版)
# ==========================================
st.set_page_config(page_title="Global Credit Lens V25.1", layout="wide", page_icon="🏦")

# CSS 样式
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
# 1. 安全鉴权
# ==========================================
def check_password():
    CORRECT_PASSWORD = "HR2026"
    if "password_correct" not in st.session_state:
        st.text_input("🔒 Access Key", type="password", key="password", on_change=lambda: st.session_state.update({"password_correct": st.session_state["password"] == CORRECT_PASSWORD}))
        return False
    return st.session_state["password_correct"]

if not check_password(): st.stop()

# ==========================================
# 2. 缓存加速
# ==========================================
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    if 'Ticker' not in df.columns: df['Ticker'] = "N/A"
    df['Ticker'] = df['Ticker'].astype(str).str.replace('.0', '', regex=False)
    return df

# ==========================================
# 3. 核心计算引擎 (Logit + Stress)
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
    st.sidebar.subheader("1. 数据接入 (Data Feed)")
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

    # B. 参数
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. 宏观压力参数")
    params = {
        'margin_shock': st.sidebar.slider("1. 行业内卷 (bps)", 0, 1000, 300),
        'tariff_shock': st.sidebar.slider("2. 关税壁垒 (%)", 0.0, 1.0, 0.25),
        'rate_hike_bps': st.sidebar.slider("3. 美联储加息 (bps)", 0, 500, 100),
        'raw_material_shock': st.sidebar.slider("4. 原材料通胀 (%)", 0, 50, 10),
        'fx_shock': st.sidebar.slider("5. 汇率风险 (%)", 0, 20, 5)
    }

    # C. 计算
    try:
        res = df_raw.apply(lambda r: CreditEngine.calculate(r, params, "Stressed"), axis=1)
        df_final = pd.concat([df_raw, res], axis=1)
        df_final['Search_Label'] = df_final['Ticker'] + " | " + df_final['Company']
    except: return

    # D. 界面
    st.title("GLOBAL CREDIT LENS | V25.1")
    st.caption(f"架构: Logit + Stress Test + IV Analysis + Smart Report")
    
    # 检索
    c_search, _ = st.columns([1, 2])
    with c_search:
        selected_label = st.selectbox("🔍 穿透式检索", df_final['Search_Label'].tolist())
    
    selected_ticker = selected_label.split(" | ")[0]
    row = df_final[df_final['Ticker'] == selected_ticker].iloc[0]

    # 单体画像
    col1, col2 = st.columns([1, 2])
    with col1:
        rating_color = '#28A745' if row['Score'] >= 70 else '#DC3545'
        st.markdown(f"""
            <div style="background-color:#1A1A1A; padding:20px; border-radius:8px; border:1px solid #333;">
                <h4 style="color:#888; margin:0;">{row['Ticker']}</h4>
                <h2 style="color:white; margin:5px 0;">{row['Company']}</h2>
                <div style="margin-top:15px; padding:10px; background-color:{rating_color}20; border-left:4px solid {rating_color};">
                    <h1 style="color:{rating_color}; margin:0; font-size:48px;">{row['Rating']}</h1>
                </div>
                <p style="color:#AAA; margin-top:10px;">Score: <b>{row['Score']:.1f}</b> | PD: <b>{row['PD_Prob']:.2%}</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        # ==========================================
        # 核心升级：V25.1 智能 PDF 报告生成引擎
        # ==========================================
        if st.button(f"📄 导出 {row['Ticker']} 深度研报"):
            try:
                # 1. 准备数据
                # 计算“基准情况” (无压力) 用于对比敏感性
                base_params = {k: 0 for k in params}
                base_res = CreditEngine.calculate(row, base_params, "Base")
                base_pd = base_res['PD_Prob']
                
                # 计算行业平均分用于对标
                avg_score = df_final['Score'].mean()
                
                # 智能建议逻辑
                if row['Score'] >= 80: action = "OUTPERFORM / BUY"
                elif row['Score'] >= 60: action = "NEUTRAL / HOLD"
                else: action = "UNDERPERFORM / SELL - REDUCE EXPOSURE"

                # 2. 生成 PDF (全英文投行格式)
                pdf = FPDF()
                pdf.add_page()
                
                # Header
                pdf.set_font("Arial", "B", 20)
                pdf.cell(0, 15, f"CREDIT MEMO: {row['Ticker']}", 0, 1)
                pdf.set_font("Arial", "", 10)
                pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
                pdf.line(10, 35, 200, 35)
                pdf.ln(5)

                # Section 1: Executive Summary
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "1. EXECUTIVE SUMMARY", 0, 1)
                pdf.set_font("Arial", "", 10)
                pdf.cell(0, 6, f"Rating: {row['Rating']} (Score: {row['Score']:.1f} / 100)", 0, 1)
                pdf.cell(0, 6, f"Industry Average Score: {avg_score:.1f}", 0, 1)
                pdf.set_font("Arial", "B", 10)
                pdf.cell(0, 6, f"Action Suggestion: {action}", 0, 1)
                pdf.ln(5)

                # Section 2: Sensitivity Analysis (Pre vs Post Stress)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "2. SENSITIVITY ANALYSIS (STRESS TEST)", 0, 1)
                pdf.set_font("Arial", "", 10)
                pdf.cell(90, 8, f"Base Case PD (No Stress):", 1)
                pdf.cell(90, 8, f"{base_pd:.2%}", 1, 1)
                pdf.set_font("Arial", "B", 10)
                pdf.cell(90, 8, f"Stressed PD (Current Scenario):", 1)
                pdf.cell(90, 8, f"{row['PD_Prob']:.2%}", 1, 1)
                pdf.ln(5)
                
                # Section 3: Financial Snapshot (The 5 Core Factors)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "3. FINANCIAL HEALTH SNAPSHOT (INPUTS)", 0, 1)
                pdf.set_font("Arial", "", 10)
                headers = ["Metric", "Value", "Risk Logic"]
                metrics = [
                    ("Gross Margin", f"{row.get('Gross Margin',0)}%", "Profitability (Survival)"),
                    ("Debt Ratio", f"{row.get('Debt Ratio',0)}%", "Solvency (Rate Sensitivity)"),
                    ("Inventory Days", f"{row.get('Inventory Days',0)} days", "Efficiency (Asset Impairment)"),
                    ("Overseas Ratio", f"{row.get('Overseas Ratio',0)}%", "Exposure (Tariff/FX Risk)"),
                    ("Cash Flow", "Positive" if row.get('Cash Flow',0)>0 else "Negative", "Liquidity (Lifeline)")
                ]
                # Table Header
                pdf.set_fill_color(240, 240, 240)
                pdf.cell(50, 8, headers[0], 1, 0, 'C', True)
                pdf.cell(40, 8, headers[1], 1, 0, 'C', True)
                pdf.cell(90, 8, headers[2], 1, 1, 'C', True)
                # Table Body
                for m, v, l in metrics:
                    pdf.cell(50, 8, m, 1)
                    pdf.cell(40, 8, v, 1)
                    pdf.cell(90, 8, l, 1, 1)
                pdf.ln(5)

                # Section 4: Stress Parameters
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "4. STRESS SCENARIO PARAMETERS", 0, 1)
                pdf.set_font("Arial", "", 9)
                pdf.cell(0, 6, f"- Margin Shock: -{params['margin_shock']} bps", 0, 1)
                pdf.cell(0, 6, f"- Tariff Shock: -{params['tariff_shock']*100:.0f}%", 0, 1)
                pdf.cell(0, 6, f"- Rate Hike: +{params['rate_hike_bps']} bps", 0, 1)
                pdf.cell(0, 6, f"- Input Cost Inflation: +{params['raw_material_shock']}%", 0, 1)
                pdf.cell(0, 6, f"- FX Depreciation: -{params['fx_shock']}%", 0, 1)
                
                pdf.ln(5)
                pdf.set_font("Arial", "I", 8)
                pdf.cell(0, 10, "Note: Automated by Global Credit Lens V25.1. Strictly for internal review.", 0, 1)
                
                pdf_bytes = bytes(pdf.output())
                st.download_button("📥 下载深度研报 (PDF)", pdf_bytes, f"Credit_Memo_{row['Ticker']}.pdf", "application/pdf")
            except Exception as e:
                st.error(f"导出失败: {e}")

    with col2:
        # 雷达图
        categories = ['综合评分', '毛利抗压', '负债健康', '现金流', '库存周转']
        def normalize(val, max_val): return min(max(val, 0), max_val) / max_val * 100
        
        row_vals = [
            row['Score'], normalize(row['Stressed_GM'] + 10, 50), normalize(100 - row['Debt Ratio'], 100),
            100 if row['Cash Flow'] > 0 else 20, normalize(365 - row['Inventory Days'], 365)
        ]
        avg_vals = [
            df_final['Score'].mean(), normalize(df_final['Stressed_GM'].mean() + 10, 50),
            normalize(100 - df_final['Debt Ratio'].mean(), 100), 60, normalize(365 - df_final['Inventory Days'].mean(), 365)
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=avg_vals, theta=categories, fill='toself', name='行业平均', line_color='#444'))
        fig.add_trace(go.Scatterpolar(r=row_vals, theta=categories, fill='toself', name=row['Company'], line_color='#00E5FF'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), template="plotly_dark", height=320, 
                          title=f"{row['Company']} 五维雷达", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=40, r=40, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # 宏观看板
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

if __name__ == "__main__":
    main()
    
