import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from fpdf import FPDF
import io

# ==========================================
# 0. 系统配置 (V30.0 Alpha Hunter)
# ==========================================
st.set_page_config(page_title="Global Credit Lens V30.0", layout="wide", page_icon="🦅")

# CSS 样式: 极客黑金 / Bloomberg 终端风格
st.markdown("""
    <style>
    /* 全局背景设为深黑 */
    .stApp { background-color: #000000 !important; color: #E0E0E0; font-family: 'Consolas', 'Roboto Mono', monospace; }
    
    /* 侧边栏 */
    [data-testid="stSidebar"] { background-color: #111 !important; border-right: 1px solid #333; }
    
    /* 字体与标题 */
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 600 !important; letter-spacing: 1px; }
    
    /* 核心指标卡片样式 */
    .stMetric { background-color: #0F0F0F; border: 1px solid #333; padding: 10px; border-radius: 0px; border-left: 3px solid #FFD700; }
    
    /* 按钮样式 (金色高亮) */
    .stButton>button { background-color: #222; color: #FFD700; border: 1px solid #FFD700; border-radius: 0px; font-weight: bold; transition: all 0.3s; }
    .stButton>button:hover { background-color: #FFD700; color: #000; }
    
    /* 输入框样式 */
    .stNumberInput input { color: #FFD700 !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 鉴权 & 数据加载
# ==========================================
def check_password():
    CORRECT_PASSWORD = "HR2026"
    if "password_correct" not in st.session_state:
        st.text_input("🔒 TERMINAL ACCESS KEY", type="password", key="password", on_change=lambda: st.session_state.update({"password_correct": st.session_state["password"] == CORRECT_PASSWORD}))
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
# 2. 核心计算引擎 (Logit + PDO Scaling)
# ==========================================
class CreditEngine:
    @staticmethod
    def sigmoid(z): return 1 / (1 + np.exp(-z))

    @staticmethod
    def scale_score(pd_val, base_score=600, base_odds=20, pdo=40):
        if pd_val >= 1.0: return 300
        if pd_val <= 0.0: return 850
        factor = pdo / np.log(2)
        offset = base_score - (factor * np.log(base_odds))
        current_odds = (1 - pd_val) / pd_val
        score = offset + (factor * np.log(current_odds))
        return int(max(300, min(850, score)))

    @staticmethod
    def calculate(row, params):
        try:
            base_gm = float(row.get('Gross Margin', 0))       
            debt_ratio = float(row.get('Debt Ratio', 50))     
            overseas = float(row.get('Overseas Ratio', 0))    
            inv = float(row.get('Inventory Days', 90))        
            cf = float(row.get('Cash Flow', 0))               
            cf_flag = 1 if cf > 0 else 0
        except: return pd.Series({'Score': 0, 'Rating': 'Error', 'PD_Prob': 1.0, 'Stressed_GM': 0})

        # 压力传导
        market_hit = params.get('margin_shock', 0) / 100.0
        tariff_hit = (overseas / 100.0) * params.get('tariff_shock', 0) * 100
        input_cost_hit = params.get('raw_material_shock', 0) * 0.2
        fx_hit = (overseas / 100.0) * params.get('fx_shock', 0) 

        final_gm = max(base_gm - market_hit - tariff_hit - input_cost_hit - fx_hit, -10.0)
        rate_hit = (debt_ratio / 100.0) * (params.get('rate_hike_bps', 0) / 100.0) * 5.0

        # Logit 回归 (Intercept = -2.0)
        logit_z = -2.0 + (-0.12 * final_gm) + (0.015 * inv) + (0.04 * debt_ratio) + (-1.5 * cf_flag) + rate_hit
        pd_val = CreditEngine.sigmoid(logit_z)
        
        # PDO 校准
        score = CreditEngine.scale_score(pd_val, base_score=600, base_odds=20, pdo=40)
        
        # 评级
        if score >= 750: rating = "AA"
        elif score >= 700: rating = "A"
        elif score >= 650: rating = "BBB"
        elif score >= 580: rating = "BB"
        elif score >= 500: rating = "B"
        else: rating = "CCC"
        
        return pd.Series({'Stressed_GM': final_gm, 'PD_Prob': pd_val, 'Score': score, 'Rating': rating})

# ==========================================
# 3. [新增] 交易阿尔法引擎 (CDS Pricing)
# ==========================================
class TradingEngine:
    """
    V30.0 核心: 将风控结果转化为交易信号
    """
    def __init__(self, recovery_rate=0.40):
        self.R = recovery_rate # 回收率 40%

    def calculate_fair_spread(self, pd_annual):
        # 简化强度模型: Spread = PD * LGD * 10000
        # LGD (违约损失率) = 1 - Recovery Rate
        spread_bps = pd_annual * (1 - self.R) * 10000
        return spread_bps

    def generate_signal(self, model_pd, market_spread_bps):
        # 1. 计算模型公允利差 (我们认为它值多少钱)
        fair_spread = self.calculate_fair_spread(model_pd)
        
        # 2. 计算 Alpha (定价偏差)
        diff = fair_spread - market_spread_bps
        threshold = 50 # 50bps 偏差才开仓
        
        # 3. 生成信号
        if diff > threshold:
            # 模型利差 > 市场利差 = 市场低估风险 = 价格太贵
            signal = "SHORT CREDIT (BUY CDS)"
            desc = f"⚠️ Risk Underpriced by {diff:.0f}bps. Arbitrage Opportunity."
            color = "#DC3545" # Red (做空/风险)
        elif diff < -threshold:
            # 模型利差 < 市场利差 = 市场过度恐慌 = 价格便宜
            signal = "LONG CREDIT (SELL CDS)"
            desc = f"💎 Value Opportunity! Mispriced by {abs(diff):.0f}bps."
            color = "#28A745" # Green (做多/机会)
        else:
            signal = "NO TRADE (HOLD)"
            desc = "Market is Efficient. No Arbitrage Gap."
            color = "#555"
            
        return fair_spread, signal, desc, color, diff

# ==========================================
# 4. 辅助引擎 (Basel, Swan, MLOps)
# ==========================================
class BaselEngine:
    def __init__(self):
        self.rw_map = {'AA': 0.2, 'A': 0.5, 'BBB': 1.0, 'BB': 1.0, 'B': 1.5, 'CCC': 1.5}
        self.capital_ratio = 0.08 
    def calculate_rwa(self, exposure, rating):
        rw = 1.5
        for key in self.rw_map:
            if rating.startswith(key):
                rw = self.rw_map[key]
                break
        return rw, exposure*rw, exposure*rw*self.capital_ratio

class BlackSwanEngine:
    @staticmethod
    def simulate_survival(row, shock_factor, fixed_cost_ratio=0.25):
        gm = float(row.get('Gross Margin', 20)) / 100.0
        base_rev = 100.0
        base_profit = base_rev - (base_rev*(1-gm)) - (base_rev*fixed_cost_ratio)
        new_rev = base_rev * (1 - shock_factor)
        new_profit = new_rev - (new_rev*(1-gm)) - (base_rev*fixed_cost_ratio)
        return {'Base_Profit': base_profit, 'Impact': new_profit-base_profit, 'Final_Profit': new_profit, 'Is_Survive': new_profit>0}

class ModelMonitor:
    @staticmethod
    def calculate_psi(expected, actual):
        try:
            breakpoints = np.nanpercentile(expected, np.linspace(0,100,11))
            e_p = np.histogram(expected, breakpoints)[0]/len(expected)
            a_p = np.histogram(actual, breakpoints)[0]/len(actual)
            e_p = np.where(e_p==0, 0.0001, e_p)
            a_p = np.where(a_p==0, 0.0001, a_p)
            return np.sum((a_p - e_p) * np.log(a_p / e_p))
        except: return 0.0

# ==========================================
# 5. 主程序
# ==========================================
def main():
    st.sidebar.title("🦅 ALPHA HUNTER TERMINAL")
    
    # 1. 数据源
    st.sidebar.caption("1. DATA FEED")
    uploaded_file = st.sidebar.file_uploader("Upload Portfolio", type=['xlsx'])
    if uploaded_file: df_raw = load_data(uploaded_file)
    else:
        df_raw = pd.DataFrame([
            {'Ticker': '600438', 'Company': '通威股份', 'Gross Margin': 28.5, 'Overseas Ratio': 25.0, 'Inventory Days': 85, 'Debt Ratio': 55.0, 'Cash Flow': 1},
            {'Ticker': '300750', 'Company': '宁德时代', 'Gross Margin': 22.0, 'Overseas Ratio': 35.0, 'Inventory Days': 70, 'Debt Ratio': 45.0, 'Cash Flow': 1},
            {'Ticker': '601012', 'Company': '隆基绿能', 'Gross Margin': 18.0, 'Overseas Ratio': 45.0, 'Inventory Days': 95, 'Debt Ratio': 50.0, 'Cash Flow': 1},
            {'Ticker': '688599', 'Company': '天合光能', 'Gross Margin': 16.0, 'Overseas Ratio': 60.0, 'Inventory Days': 80, 'Debt Ratio': 65.0, 'Cash Flow': 1},
            {'Ticker': '002459', 'Company': '晶澳科技', 'Gross Margin': 15.5, 'Overseas Ratio': 55.0, 'Inventory Days': 88, 'Debt Ratio': 60.0, 'Cash Flow': 0}
        ])

    # 2. 宏观参数
    st.sidebar.caption("2. MACRO SHOCKS")
    params = {
        'margin_shock': st.sidebar.slider("Margin Squeeze (bps)", 0, 1000, 300),
        'tariff_shock': st.sidebar.slider("Tariff (%)", 0.0, 1.0, 0.25),
        'rate_hike_bps': st.sidebar.slider("Rate Hike (bps)", 0, 500, 100),
        'raw_material_shock': st.sidebar.slider("Input Inflation (%)", 0, 50, 10),
        'fx_shock': st.sidebar.slider("FX Impact (%)", 0, 20, 5)
    }
    
    # 模拟计算
    try:
        res = df_raw.apply(lambda r: CreditEngine.calculate(r, params), axis=1)
        df_final = pd.concat([df_raw, res], axis=1)
        df_final['Search_Label'] = df_final['Ticker'] + " | " + df_final['Company']
    except: return

    # 3. MLOps 监控
    np.random.seed(42)
    # 模拟训练集 (Benchmark)
    train_scores = np.random.normal(700, 50, 1000)
    # 计算当前 PSI
    psi = ModelMonitor.calculate_psi(train_scores, df_final['Score'].values)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("3. MODEL HEALTH (MLOps)")
    st.sidebar.metric("PSI Monitor", f"{psi:.3f}", delta="Stable" if psi<0.1 else "Drift Detected", delta_color="inverse")
    if psi > 0.1: st.sidebar.warning("⚠️ Data Drift Alert!")

    # ==========================================
    # Alpha Hunter 主界面
    # ==========================================
    st.title("GLOBAL CREDIT LENS | V30.0")
    st.caption("Mode: Distressed Alpha Hunter | Strategy: CDS Arbitrage")

    # 资产选择
    c_search, _ = st.columns([1, 2])
    with c_search:
        selected_label = st.selectbox("🎯 TARGET ASSET", df_final['Search_Label'].tolist())
    
    selected_ticker = selected_label.split(" | ")[0]
    row = df_final[df_final['Ticker'] == selected_ticker].iloc[0]

    # --- 核心模块: 交易控制台 (Trading Desk) ---
    st.markdown("### 📡 ALPHA TRADING DESK")
    
    # 模拟市场数据输入 (交易员操作区)
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        market_spread = st.number_input("📉 Market CDS Spread (bps)", value=300, step=10, help="当前市场上该公司的信用违约互换报价")
    with c2:
        recovery = st.number_input("♻️ Recovery Rate (%)", value=40, step=5) / 100.0
    
    # 调用 Alpha 引擎生成信号
    trader = TradingEngine(recovery_rate=recovery)
    fair_spread, signal, desc, color, diff = trader.generate_signal(row['PD_Prob'], market_spread)

    with c3:
        # 信号展示卡片
        st.markdown(f"""
            <div style="background-color:#111; padding:15px; border: 1px solid {color}; border-left: 10px solid {color};">
                <h2 style="color:{color}; margin:0; font-family:'Arial Black'; letter-spacing:1px;">{signal}</h2>
                <p style="color:#EEE; font-size:16px; margin:5px 0;">{desc}</p>
                <p style="color:#888; font-size:12px; margin:0;">Model Fair Value: <b>{fair_spread:.0f} bps</b> vs Market: <b>{market_spread:.0f} bps</b></p>
            </div>
        """, unsafe_allow_html=True)

    # --- 仪表盘可视化 ---
    col1, col2 = st.columns([1, 1])
    with col1:
        # Spread Gap Gauge (套利空间)
        fig = go.Figure(go.Indicator(
            mode = "number+delta",
            value = fair_spread,
            delta = {'reference': market_spread, 'position': "top", 'valueformat': ".0f"},
            title = {'text': f"Arbitrage Spread Gap (bps)", 'font': {'size': 14, 'color': '#888'}},
            number = {'suffix': " bps", 'font': {'size': 50, 'color': 'white'}},
            domain = {'row': 0, 'column': 0}
        ))
        fig.update_layout(height=200, margin=dict(t=30,b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Credit Score Gauge (风控基础)
        fig_score = go.Figure(go.Indicator(
            mode = "gauge+number", value = row['Score'],
            title = {'text': f"Credit Score (PD: {row['PD_Prob']:.1%})", 'font': {'size': 14, 'color': '#888'}},
            gauge = {'axis': {'range': [300, 850]}, 'bar': {'color': color}, 'bgcolor': "#222", 
                     'steps': [{'range': [300,550], 'color':'#300'}, {'range': [650,850], 'color':'#030'}]}
        ))
        fig_score.update_layout(height=200, margin=dict(t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', font={'color':'white'})
        st.plotly_chart(fig_score, use_container_width=True)

    # --- 辅助模块 (Basel & Swan) ---
    st.markdown("---")
    st.subheader("🛠️ RISK & CAPITAL ANALYTICS")
    
    # 资本 & 黑天鹅计算
    basel = BaselEngine()
    _, _, cap_stress = basel.calculate_rwa(10_000_000, row['Rating'])
    swan = BlackSwanEngine.simulate_survival(row, 0.4, 0.25) # 默认40%冲击
    
    bc1, bc2 = st.columns(2)
    with bc1:
        st.metric("Basel III Capital Charge", f"${cap_stress:,.0f}", "Stressed RWA Impact", delta_color="inverse")
    with bc2:
        st.metric("Black Swan Survival", "SURVIVED" if swan['Is_Survive'] else "BANKRUPT", f"Profit Impact: {swan['Impact']:.1f}", delta_color="normal" if swan['Is_Survive'] else "inverse")

    # 导出 Alpha 策略报告
    if st.button("📄 Generate Alpha Strategy Memo"):
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, f"ALPHA STRATEGY MEMO: {row['Ticker']}", 0, 1)
            pdf.line(10, 20, 200, 20)
            pdf.ln(5)
            
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
            pdf.cell(0, 10, f"Target: {row['Company']}", 0, 1)
            
            pdf.set_font("Arial", "B", 14)
            pdf.ln(5)
            pdf.cell(0, 10, "TRADING SIGNAL", 0, 1)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Signal: {signal}", 0, 1)
            pdf.cell(0, 10, f"Recommendation: {desc}", 0, 1)
            
            pdf.set_font("Arial", "B", 14)
            pdf.ln(5)
            pdf.cell(0, 10, "PRICING FUNDAMENTALS", 0, 1)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Model PD: {row['PD_Prob']:.2%}", 0, 1)
            pdf.cell(0, 10, f"Fair Spread (Model): {fair_spread:.0f} bps", 0, 1)
            pdf.cell(0, 10, f"Market Spread: {market_spread:.0f} bps", 0, 1)
            pdf.cell(0, 10, f"Arbitrage Gap: {diff:.0f} bps", 0, 1)
            
            st.download_button("📥 Download PDF", bytes(pdf.output()), f"Alpha_Memo_{row['Ticker']}.pdf")
        except: st.error("PDF Generation Error")

if __name__ == "__main__":
    main()
