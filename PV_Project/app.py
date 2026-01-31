import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from fpdf import FPDF
import io

# ==========================================
# 0. 系统配置 (V31.2 Stable Fix Edition)
# ==========================================
st.set_page_config(page_title="Global Credit Lens V31.2", layout="wide", page_icon="🦅")

st.markdown("""
    <style>
    .stApp { background-color: #000000 !important; color: #E0E0E0; font-family: 'Consolas', 'Roboto Mono', monospace; }
    [data-testid="stSidebar"] { background-color: #111 !important; border-right: 1px solid #333; }
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 600 !important; }
    .stMetric { background-color: #0F0F0F; border: 1px solid #333; padding: 10px; border-radius: 0px; border-left: 3px solid #FFD700; }
    .stButton>button { background-color: #222; color: #FFD700; border: 1px solid #FFD700; border-radius: 0px; font-weight: bold; width: 100%; }
    .stButton>button:hover { background-color: #FFD700; color: #000; }
    .streamlit-expanderHeader { background-color: #222 !important; color: #FFF !important; }
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
# 2. 核心计算引擎 (Logit + PDO)
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
    def calculate(row, params, calibration_intercept):
        try:
            base_gm = float(row.get('Gross Margin', 0))       
            debt_ratio = float(row.get('Debt Ratio', 50))     
            overseas = float(row.get('Overseas Ratio', 0))    
            inv = float(row.get('Inventory Days', 90))        
            cf = float(row.get('Cash Flow', 0))               
            cf_flag = 1 if cf > 0 else 0
        except: return pd.Series({'Score': 0, 'Rating': 'Error', 'PD_Prob': 1.0, 'Stressed_GM': 0})

        market_hit = params.get('margin_shock', 0) / 100.0
        tariff_hit = (overseas / 100.0) * params.get('tariff_shock', 0) * 100
        input_cost_hit = params.get('raw_material_shock', 0) * 0.2
        fx_hit = (overseas / 100.0) * params.get('fx_shock', 0) 

        final_gm = max(base_gm - market_hit - tariff_hit - input_cost_hit - fx_hit, -10.0)
        rate_hit = (debt_ratio / 100.0) * (params.get('rate_hike_bps', 0) / 100.0) * 5.0

        logit_z = calibration_intercept + (-0.12 * final_gm) + (0.015 * inv) + (0.04 * debt_ratio) + (-1.5 * cf_flag) + rate_hit
        pd_val = CreditEngine.sigmoid(logit_z)
        score = CreditEngine.scale_score(pd_val, base_score=600, base_odds=20, pdo=40)
        
        if score >= 750: rating = "AA"
        elif score >= 700: rating = "A"
        elif score >= 650: rating = "BBB"
        elif score >= 580: rating = "BB"
        elif score >= 500: rating = "B"
        else: rating = "CCC"
        
        return pd.Series({'Stressed_GM': final_gm, 'PD_Prob': pd_val, 'Score': score, 'Rating': rating})

# ==========================================
# 3. 交易引擎 (CDS Alpha)
# ==========================================
class TradingEngine:
    def __init__(self, recovery_rate=0.40):
        self.R = recovery_rate

    def calculate_fair_spread(self, pd_annual):
        return pd_annual * (1 - self.R) * 10000

    def generate_signal(self, model_pd, market_spread_bps):
        fair_spread = self.calculate_fair_spread(model_pd)
        diff = fair_spread - market_spread_bps
        threshold = 50 
        
        if diff > threshold:
            signal = "SHORT CREDIT (BUY CDS)"
            desc = f"Risk Underpriced by {diff:.0f}bps"
            color = "#DC3545" 
        elif diff < -threshold:
            signal = "LONG CREDIT (SELL CDS)"
            desc = f"Value Opportunity! Mispriced by {abs(diff):.0f}bps"
            color = "#28A745" 
        else:
            signal = "HOLD"
            desc = "Market is Efficient"
            color = "#555"
            
        return fair_spread, signal, desc, color, diff

# ==========================================
# 4. NLP 舆情引擎 (English Mock Data)
# ==========================================
class SentimentEngine:
    @staticmethod
    def analyze_news(ticker):
        # [关键修复] 使用英文 Mock 数据，防止 PDF 生成崩溃
        # 如果这里使用 AkShare 抓取中文，PDF 生成会报错，除非安装中文字体
        news_db = {
            '300750': [("Reuters", "CATL confirms new plant deal with Ford in US", "Positive"), ("Bloomberg", "Lithium prices drop 20%, boosting margins", "Positive")],
            '601012': [("WSJ", "LONGi solar panels detained by US Customs", "Negative"), ("Analyst", "Global oversupply warning issued", "Negative")],
            '600438': [("Financial Times", "Tongwei expands polysilicon capacity", "Positive")],
            '688599': [("MarketWatch", "Trina Solar faces new tariff probe", "Negative")],
            '002459': [("Reuters", "JA Solar reports steady Q3 growth", "Positive")]
        }
        default_news = [("AI Feed", "No major sentiment shift detected", "Neutral")]
        news_list = news_db.get(ticker, default_news)
        
        score = 0
        for _, _, sentiment in news_list:
            if sentiment == "Positive": score += 10
            elif sentiment == "Negative": score -= 10
        final_score = max(-100, min(100, score * 5))
        
        if final_score > 20: label = "Bullish"
        elif final_score < -20: label = "Bearish"
        else: label = "Neutral"
        
        return news_list, final_score, label

# ==========================================
# 5. 辅助引擎 (Basel, Swan, MLOps, IV)
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
# 6. PDF 生成器 (Fix: Chinese Mapping & UTF8)
# ==========================================
def generate_pdf_report(row, signal, fair_spread, market_spread, diff, cap_stress, swan, sent_score, sent_label, news_list):
    # [关键修复] 中文映射表：解决 FPDF 无法显示中文的问题
    # 强制将中文公司名转换为英文，确保 PDF 生成成功
    company_map = {
        '通威股份': 'Tongwei Co.',
        '宁德时代': 'CATL',
        '隆基绿能': 'LONGi Green Energy',
        '天合光能': 'Trina Solar',
        '晶澳科技': 'JA Solar Technology'
    }
    # 如果找不到映射，就用 Ticker 代替，绝不让它报错
    company_name_en = company_map.get(row['Company'], f"Ticker {row['Ticker']}")

    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"INSTITUTIONAL MEMO: {row['Ticker']}", 0, 1)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | System: Global Credit Lens V31.2", 0, 1)
    pdf.line(10, 25, 200, 25)
    pdf.ln(5)
    
    # 1. Executive Summary
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "1. EXECUTIVE SUMMARY", 0, 1)
    pdf.set_font("Arial", "", 10)
    # [关键] 使用映射后的英文名
    pdf.cell(0, 8, f"Target: {company_name_en}", 0, 1) 
    pdf.cell(0, 8, f"Rating: {row['Rating']} (Score: {row['Score']})", 0, 1)
    pdf.cell(0, 8, f"Alpha Signal: {signal}", 0, 1)
    pdf.ln(5)

    # 2. Trading Analytics
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "2. ALPHA TRADING DESK", 0, 1)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, f"Market CDS Spread: {market_spread:.0f} bps", 0, 1)
    pdf.cell(0, 8, f"Model Fair Value: {fair_spread:.0f} bps", 0, 1)
    pdf.cell(0, 8, f"Arbitrage Gap: {diff:.0f} bps", 0, 1)
    pdf.ln(5)

    # 3. Sentiment Radar
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "3. SENTIMENT RADAR (NLP)", 0, 1)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, f"Sentiment Score: {sent_score} ({sent_label})", 0, 1)
    pdf.cell(0, 8, "Latest Headlines:", 0, 1)
    pdf.set_font("Arial", "I", 8)
    for src, title, tag in news_list:
        # News titles are English (from mock data)
        pdf.cell(0, 6, f"  - [{src}] {title} ({tag})", 0, 1)
    pdf.ln(5)
    
    # 4. Risk & Capital
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "4. RISK & CAPITAL", 0, 1)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, f"Basel III Capital Charge: ${cap_stress:,.0f}", 0, 1)
    pdf.cell(0, 8, f"Black Swan Survival: {'YES' if swan['Is_Survive'] else 'NO'}", 0, 1)

    return bytes(pdf.output())

# ==========================================
# 7. 主程序
# ==========================================
def main():
    st.sidebar.title("🦅 ALPHA HUNTER")
    
    # 1. Inputs
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

    st.sidebar.caption("2. PARAMETERS")
    params = {
        'margin_shock': st.sidebar.slider("Margin Squeeze", 0, 1000, 300),
        'tariff_shock': st.sidebar.slider("Tariff", 0.0, 1.0, 0.25),
        'rate_hike_bps': st.sidebar.slider("Rate Hike", 0, 500, 100),
        'raw_material_shock': st.sidebar.slider("Inflation", 0, 50, 10),
        'fx_shock': st.sidebar.slider("FX", 0, 20, 5)
    }
    calib_intercept = st.sidebar.slider("Calibration (Intercept)", -5.0, 2.0, -1.0)

    try:
        res = df_raw.apply(lambda r: CreditEngine.calculate(r, params, calib_intercept), axis=1)
        df_final = pd.concat([df_raw, res], axis=1)
        df_final['Search_Label'] = df_final['Ticker'] + " | " + df_final['Company']
    except: return

    # MLOps
    np.random.seed(42)
    psi = ModelMonitor.calculate_psi(np.random.normal(700,50,1000), df_final['Score'].values)
    st.sidebar.markdown("---")
    st.sidebar.metric("MLOps: PSI Monitor", f"{psi:.3f}", delta="Stable" if psi<0.1 else "Drift", delta_color="inverse")

    # Main UI
    st.title("GLOBAL CREDIT LENS | V31.2")
    st.caption("Mode: Distressed Alpha Hunter (Stable Edition)")

    c_search, _ = st.columns([1, 2])
    with c_search:
        selected_label = st.selectbox("🎯 TARGET ASSET", df_final['Search_Label'].tolist())
    
    selected_ticker = selected_label.split(" | ")[0]
    row = df_final[df_final['Ticker'] == selected_ticker].iloc[0]

    # --- Trading Desk ---
    st.markdown("### 📡 ALPHA TRADING DESK")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1: market_spread = st.number_input("📉 Market CDS (bps)", value=300, step=10)
    with c2: recovery = st.number_input("♻️ Recovery (%)", value=40, step=5) / 100.0
    
    trader = TradingEngine(recovery)
    fair_spread, signal, desc, color, diff = trader.generate_signal(row['PD_Prob'], market_spread)

    with c3:
        st.markdown(f"""
            <div style="background-color:#111; padding:15px; border: 1px solid {color}; border-left: 10px solid {color};">
                <h2 style="color:{color}; margin:0;">{signal}</h2>
                <p style="color:#EEE;">{desc}</p>
            </div>
        """, unsafe_allow_html=True)
        
    # --- NLP Radar ---
    st.markdown("### 📰 NLP SENTIMENT RADAR")
    nlp = SentimentEngine()
    news_list, sent_score, sent_label = nlp.analyze_news(row['Ticker'])
    
    nc1, nc2 = st.columns([1, 3])
    with nc1:
        sent_color = "#28A745" if sent_score > 0 else ("#DC3545" if sent_score < 0 else "#888")
        st.metric("AI Sentiment Score", sent_score, sent_label)
    with nc2:
        for src, title, tag in news_list:
            t_col = "green" if tag=="Positive" else ("red" if tag=="Negative" else "grey")
            st.markdown(f"<span style='color:#FFD700'>[{src}]</span> {title} <span style='color:{t_col}'>({tag})</span>", unsafe_allow_html=True)

    # --- Analytics Charts ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Indicator(mode="number+delta", value=fair_spread, delta={'reference': market_spread}, title={'text': "Arbitrage Gap (bps)"}))
        fig.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', font={'color':'white'})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig_score = go.Figure(go.Indicator(mode="gauge+number", value=row['Score'], gauge={'axis': {'range': [300, 850]}, 'bar': {'color': color}}))
        fig_score.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', font={'color':'white'})
        st.plotly_chart(fig_score, use_container_width=True)

    # --- Risk Analytics (Basel & Swan) ---
    basel = BaselEngine()
    _, _, cap_stress = basel.calculate_rwa(10_000_000, row['Rating'])
    swan = BlackSwanEngine.simulate_survival(row, 0.4, 0.25)
    
    tab1, tab2, tab3 = st.tabs(["🏛️ Basel Capital", "🏴‍☠️ Black Swan", "🛁 Competition"])
    with tab1:
        # [修复] 找回三步瀑布图
        fig_cap = go.Figure(go.Waterfall(measure=["relative", "relative", "total"], x=["Base", "Impact", "Final"], y=[0, cap_stress, cap_stress], text=[f"${cap_stress/1000:.0f}k", f"+${0:.0f}k", f"${cap_stress/1000:.0f}k"], textfont=dict(color="white"), connector={"line":{"color":"#666"}}, decreasing={"marker":{"color":"#FF4B4B"}}, totals={"marker":{"color":"#EEE"}}))
        fig_cap.update_layout(title="Capital Impact", template="plotly_dark", height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cap, use_container_width=True)
    with tab2:
        is_alive = swan['Is_Survive']
        fig_swan = go.Figure(go.Waterfall(measure=["relative", "relative", "total"], x=["Base", "Shock", "Final"], y=[swan['Base_Profit'], swan['Impact'], swan['Final_Profit']], text=[f"{swan['Base_Profit']:.1f}", f"{swan['Impact']:.1f}", f"{swan['Final_Profit']:.1f}"], textfont=dict(color="white"), connector={"line":{"color":"#666"}}, increasing={"marker":{"color":"#28A745"}}, decreasing={"marker":{"color":"#FF4B4B"}}, totals={"marker":{"color": "#FFF" if is_alive else "#555"}}))
        fig_swan.update_layout(title="Survival Test", template="plotly_dark", height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_swan, use_container_width=True)
    with tab3:
        if not df_final.empty:
            st.plotly_chart(px.scatter(df_final, x="Stressed_GM", y="Score", size="Debt Ratio", color="Rating", hover_name="Company", title="Profit vs Risk"), use_container_width=True)

    # --- Footer & PDF Download ---
    st.markdown("---")
    
    # PDF 生成与下载
    pdf_bytes = generate_pdf_report(row, signal, fair_spread, market_spread, diff, cap_stress, swan, sent_score, sent_label, news_list)
    
    st.download_button(
        label="📥 DOWNLOAD INSTITUTIONAL MEMO (PDF)",
        data=pdf_bytes,
        file_name=f"Alpha_Memo_{row['Ticker']}.pdf",
        mime="application/pdf",
        help="Generate comprehensive PDF report with Risk, Capital, and Trading signals."
    )
    
    with st.expander("🏗️ System Architecture (V31.2)"):
        st.markdown("**Core:** Logit+PDO / CDS Pricing / Swan Engine / NLP Sentiment / MLOps PSI")

if __name__ == "__main__":
    main()
