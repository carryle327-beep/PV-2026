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
# 0. 系统配置 (V24.2 修复版)
# ==========================================
st.set_page_config(page_title="Global Credit Lens V24.2", layout="wide", page_icon="🏦")

# [修复]：删除了报错的 set_option('deprecation.showPyplotGlobalUse', False)
# 因为代码里主要用 Plotly，这行旧配置已不再需要。

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
    def password_entered():
        if st.session_state["password"] == CORRECT_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔒 访问密钥 (Access Key)", type="password", on_change=password_entered, key="password")
        st.caption("提示: HR2026")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔒 访问密钥 (Access Key)", type="password", on_change=password_entered, key="password")
        st.error("⛔ 密钥错误")
        return False
    else:
        return True

if not check_password():
    st.stop()

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
# 3. 五维计算引擎 (5-Factor Engine)
# ==========================================
class CreditEngine:
    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -10, 10)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def calculate(row, params, macro_status):
        try:
            base_gm = float(row.get('Gross Margin', 0))
            debt_ratio = float(row.get('Debt Ratio', 50))
            overseas = float(row.get('Overseas Ratio', 0))
            inv = float(row.get('Inventory Days', 90))
            cf = float(row.get('Cash Flow', 0))
            cf_flag = 1 if cf > 0 else 0
        except:
            return pd.Series({'Score': 0, 'Rating': 'Error', 'PD_Prob': 1.0, 'Stressed_GM': 0})

        # --- 五维压力传导逻辑 ---
        
        # 1. 行业内卷 (Margin Shock)
        market_hit = params['margin_shock'] / 100.0
        
        # 2. 关税冲击 (Tariff Shock)
        tariff_hit = (overseas / 100.0) * params['tariff_shock'] * 100
        
        # 3. 原材料通胀 (Input Cost)
        input_cost_hit = params['raw_material_shock'] * 0.2
        
        # 4. 汇率冲击 (FX Shock)
        fx_hit = (overseas / 100.0) * params['fx_shock'] 

        # 计算折后毛利
        final_gm = base_gm - market_hit - tariff_hit - input_cost_hit - fx_hit
        final_gm = max(final_gm, -10.0)

        # 5. 加息冲击 (Rate Hike)
        rate_hit = (debt_ratio / 100.0) * (params['rate_hike_bps'] / 100.0) * 5.0

        # --- Logit 模型 ---
        intercept = -0.5
        logit_z = intercept + \
                  (-0.15 * final_gm) + \
                  (0.02 * inv) + \
                  (0.05 * debt_ratio) + \
                  (-1.2 * cf_flag) + \
                  rate_hit
                  
        pd_val = CreditEngine.sigmoid(logit_z)
        score = 100 * (1 - pd_val)
        
        # 评级
        if score >= 85: rating = "AAA"
        elif score >= 70: rating = "AA"
        elif score >= 55: rating = "BBB"
        elif score >= 40: rating = "BB"
        else: rating = "CCC"
        
        return pd.Series({
            'Stressed_GM': final_gm, 'PD_Prob': pd_val, 'Score': score, 'Rating': rating
        })

# ==========================================
# 4. 主程序
# ==========================================
def main():
    st.sidebar.title("🎛️ 压力测试实验室")
    
    # --- A. 数据源 ---
    st.sidebar.subheader("1. 数据接入")
    uploaded_file = st.sidebar.file_uploader("上传 Excel", type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            df_raw = load_data(uploaded_file)
            st.sidebar.success(f"已联网: {len(df_raw)} 家主体")
        except: return
    else:
        st.sidebar.info("使用演示数据...")
        df_raw = pd.DataFrame([
            {'Ticker': '600438', 'Company': '通威股份', 'Gross Margin': 28.5, 'Overseas Ratio': 25.0, 'Inventory Days': 85, 'Debt Ratio': 55.0, 'Cash Flow': 1},
            {'Ticker': '300750', 'Company': '宁德时代', 'Gross Margin': 22.0, 'Overseas Ratio': 35.0, 'Inventory Days': 70, 'Debt Ratio': 45.0, 'Cash Flow': 1}
        ])

    # --- B. 五维压力参数 (5 Factors) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. 宏观压力参数 (5 Factors)")
    
    st.sidebar.caption("📉 市场环境")
    margin_shock = st.sidebar.slider("1. 行业内卷 (bps)", 0, 1000, 300)
    
    st.sidebar.caption("🚢 地缘政治")
    tariff_shock = st.sidebar.slider("2. 关税壁垒 (%)", 0.0, 1.0, 0.25)
    
    st.sidebar.caption("💰 资金成本")
    rate_hike = st.sidebar.slider("3. 美联储加息 (bps)", 0, 500, 100)
    
    st.sidebar.caption("🧱 供应链")
    raw_mat_shock = st.sidebar.slider("4. 原材料通胀 (%)", 0, 50, 10)
    
    st.sidebar.caption("💱 汇率风险")
    fx_shock = st.sidebar.slider("5. 汇率波动损失 (%)", 0, 20, 5)
    
    params = {
        'margin_shock': margin_shock, 
        'tariff_shock': tariff_shock,
        'rate_hike_bps': rate_hike,
        'raw_material_shock': raw_mat_shock,
        'fx_shock': fx_shock
    }

    # --- 计算 ---
    try:
        res = df_raw.apply(lambda r: CreditEngine.calculate(r, params, "衰退期"), axis=1)
        df_final = pd.concat([df_raw, res], axis=1)
        df_final['Search_Label'] = df_final['Ticker'] + " | " + df_final['Company']
    except: return

    # --- 界面 ---
    st.title("GLOBAL CREDIT LENS | V24.2")
    st.caption(f"模型状态: 五维全量压力测试 (5-Factor Stress Model) | 修复版")
    
    # 搜索
    search_list = df_final['Search_Label'].tolist()
    c_search, c_blank = st.columns([1, 2])
    with c_search:
        selected_label = st.selectbox("🔍 穿透式检索 (Ticker/Name)", search_list)
    
    selected_ticker = selected_label.split(" | ")[0]
    row = df_final[df_final['Ticker'] == selected_ticker].iloc[0]

    # 单体
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
        
        # PDF 导出 (包含 5 个参数)
        if st.button(f"📄 导出 {row['Ticker']} 研报"):
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 24)
                pdf.cell(0, 20, f"CREDIT MEMO: {row['Ticker']}", 0, 1, 'C')
                pdf.line(10, 30, 200, 30)
                pdf.ln(10)
                
                pdf.set_font("Arial", "", 12)
                pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
                pdf.cell(0, 10, f"Rating: {str(row['Rating']).split(' ')[0]}", 0, 1)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"PD: {row['PD_Prob']:.2%}", 0, 1)
                
                pdf.ln(10)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "5-FACTOR STRESS TEST:", 0, 1)
                pdf.set_font("Arial", "", 11)
                pdf.cell(0, 8, f"1. Margin Shock: -{params['margin_shock']} bps", 0, 1)
                pdf.cell(0, 8, f"2. Tariff Shock: -{params['tariff_shock']*100:.0f}%", 0, 1)
                pdf.cell(0, 8, f"3. Rate Hike: +{params['rate_hike_bps']} bps", 0, 1)
                pdf.cell(0, 8, f"4. Input Cost: +{params['raw_material_shock']}%", 0, 1)
                pdf.cell(0, 8, f"5. FX Shock: -{params['fx_shock']}%", 0, 1)
                
                pdf.ln(10)
                pdf.set_font("Arial", "I", 10)
                pdf.cell(0, 10, "Note: Automated by Global Credit Lens V24.2", 0, 1)
                
                pdf_bytes = bytes(pdf.output())
                st.download_button("📥 下载 PDF", pdf_bytes, f"Report_{row['Ticker']}.pdf", "application/pdf")
            except Exception as e:
                st.error(f"导出失败: {e}")

    with col2:
        # 雷达图
        categories = ['综合评分', '毛利抗压', '负债健康', '现金流', '库存周转']
        def normalize(val, max_val): return min(max(val, 0), max_val) / max_val * 100
        
        row_vals = [
            row['Score'], 
            normalize(row['Stressed_GM'] + 10, 50), 
            normalize(100 - row['Debt Ratio'], 100),
            100 if row['Cash Flow'] > 0 else 20,
            normalize(365 - row['Inventory Days'], 365)
        ]
        
        avg_vals = [
            df_final['Score'].mean(),
            normalize(df_final['Stressed_GM'].mean() + 10, 50),
            normalize(100 - df_final['Debt Ratio'].mean(), 100),
            60,
            normalize(365 - df_final['Inventory Days'].mean(), 365)
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=avg_vals, theta=categories, fill='toself', name='行业平均', line_color='#444'))
        fig.add_trace(go.Scatterpolar(r=row_vals, theta=categories, fill='toself', name=row['Company'], line_color='#00E5FF'))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            template="plotly_dark", height=320, 
            title=f"{row['Company']} 五维雷达",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 深度量化看板")
    tab1, tab2, tab3, tab4 = st.tabs(["全景热力", "竞争气泡", "评级分布", "因子归因"])
    
    with tab1:
        if not df_final.empty:
            fig_map = px.treemap(df_final, path=[px.Constant("全市场"), 'Rating', 'Search_Label'], values='Score',
                                 color='Score', color_continuous_scale='RdYlGn')
            fig_map.update_layout(template="plotly_dark", height=450, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_map, use_container_width=True)
            
    with tab2:
        if not df_final.empty:
            fig_bub = px.scatter(df_final, x="Stressed_GM", y="Score", size="Debt Ratio", color="Rating",
                                 hover_name="Company", text="Company",
                                 color_discrete_sequence=px.colors.qualitative.Bold)
            fig_bub.update_layout(template="plotly_dark", height=450, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bub, use_container_width=True)

    with tab3:
        if not df_final.empty:
            fig_vio = px.strip(df_final, x="Rating", y="Score", color="Rating")
            fig_vio.update_layout(template="plotly_dark", height=450, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_vio, use_container_width=True)

    with tab4:
        if not df_final.empty:
            cols_to_corr = ['Score', 'Gross Margin', 'Overseas Ratio', 'Inventory Days', 'Debt Ratio']
            corr_matrix = df_final[cols_to_corr].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
            fig_corr.update_layout(template="plotly_dark", height=450, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_corr, use_container_width=True)

if __name__ == "__main__":
    main()
