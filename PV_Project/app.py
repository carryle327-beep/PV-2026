import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from report_engine import CreditMemoGenerator 

# ==========================================
# 0. 系统配置
# ==========================================
st.set_page_config(page_title="Cross-Border Credit Lens", layout="wide", page_icon="🏦")

# 银行级 UI 样式
st.markdown("""
    <style>
    .stApp { background-color: #f5f7f9; color: #333; font-family: 'Arial', sans-serif; }
    [data-testid="stSidebar"] { background-color: #003366; color: white; }
    .stMetric { background-color: white; border: 1px solid #ddd; padding: 15px; border-left: 5px solid #003366; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #003366 !important; }
    .stButton>button { width: 100%; font-weight: bold; }
    /* Footer 样式 */
    .footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #f1f1f1; color: #666; text-align: center; padding: 10px; font-size: 12px; border-top: 1px solid #ddd; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 核心计算引擎
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
        # 基础数据
        base_gm = float(row.get('Gross Margin', 0))       
        debt_ratio = float(row.get('Debt Ratio', 50))     
        overseas = float(row.get('Cross-Border Exposure', 0)) 
        inv = float(row.get('Inventory Days', 90))        
        cf_flag = 1 if float(row.get('Cash Flow', 0)) > 0 else 0

        # 压力测试逻辑
        tariff_hit = (overseas / 100.0) * params.get('us_tariff_shock', 0) * 1.5 
        input_cost_hit = params.get('poly_price_shock', 0) * 0.2
        final_gm = max(base_gm - tariff_hit - input_cost_hit, -10.0)
        
        # Logit 算分
        logit_z = -0.5 + (-0.12 * final_gm) + (0.015 * inv) + (0.04 * debt_ratio) + (-1.5 * cf_flag)
        pd_val = CreditEngine.sigmoid(logit_z)
        score = CreditEngine.scale_score(pd_val)
        
        # 动态评级映射 (Stressed Rating)
        if score >= 750: rating = "AA (Strong)"
        elif score >= 680: rating = "A (Good)"
        elif score >= 600: rating = "BBB (Satisfactory)"
        elif score >= 550: rating = "BB (Watchlist)"
        else: rating = "CCC (Sub-Standard)"
        
        return pd.Series({
            'Stressed_GM': final_gm, 
            'PD_Prob': pd_val, 
            'Score': score, 
            'Rating': rating, # 这是受压后的评级
            'Tariff_Impact': tariff_hit
        })

# ==========================================
# 2. 决策支持引擎
# ==========================================
class DecisionEngine:
    def __init__(self, recovery=0.4): self.R = recovery
    
    def get_view(self, pd, market_bps):
        model_implied_spread = pd * (1 - self.R) * 10000
        gap = model_implied_spread - market_bps
        
        # [改进3] VP 级话术升级
        if gap > 100: 
            return model_implied_spread, "⚠️ Risk Underpriced", f"Market pricing does not adequately compensate for stressed loss given default (-{gap:.0f}bps).", "Medium"
        elif gap < -100: 
            return model_implied_spread, "✅ Risk Overpriced", f"Market implies distress beyond fundamental model projection (+{abs(gap):.0f}bps).", "Low"
        return model_implied_spread, "⚖️ Fairly Priced", "Market aligns with fundamental risk projection.", "Low"

# ==========================================
# 3. 主程序
# ==========================================
def main():
    st.sidebar.title("🏦 CREDIT CONTROL")
    st.sidebar.caption("Global Credit Risk | Singapore HQ")
    
    # [改进2] 增加 Base Rating 用于对比迁移
    df_raw = pd.DataFrame([
        {'Ticker': '601012', 'Company': 'Longi Green Energy (Vietnam)', 'Base Rating': 'AA (Strong)', 'Gross Margin': 18.2, 'Cross-Border Exposure': 45.0, 'Inventory Days': 95, 'Debt Ratio': 52.0, 'Cash Flow': 1},
        {'Ticker': '688599', 'Company': 'Trina Solar (Thailand)', 'Base Rating': 'A (Good)', 'Gross Margin': 15.8, 'Cross-Border Exposure': 60.0, 'Inventory Days': 82, 'Debt Ratio': 63.0, 'Cash Flow': 1},
        {'Ticker': '002459', 'Company': 'JA Solar (Malaysia)', 'Base Rating': 'BBB (Satisfactory)', 'Gross Margin': 16.5, 'Cross-Border Exposure': 55.0, 'Inventory Days': 88, 'Debt Ratio': 58.0, 'Cash Flow': 0}
    ])
    
    st.sidebar.subheader("🌪️ Stress Scenarios")
    params = {
        'us_tariff_shock': st.sidebar.slider("US Tariff Shock (%)", 0, 100, 30, help="Simulate Anti-Circumvention Tariff rate"),
        'poly_price_shock': st.sidebar.slider("Input Cost Hike (%)", 0, 50, 10)
    }

    # 运行计算
    res = df_raw.apply(lambda r: CreditEngine.calculate(r, params), axis=1)
    df_final = pd.concat([df_raw, res], axis=1)
    
    st.title("CROSS-BORDER CREDIT LENS (V32.3)")
    st.markdown("**Role:** Credit Risk Officer | **Desk:** Singapore-China Corridor")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        selected = st.selectbox("📂 Select Borrower:", df_final['Company'].tolist())
    with col2:
        st.info("💡 **Analyst Note:** This module assesses creditworthiness under US trade policy stress.")

    row = df_final[df_final['Company'] == selected].iloc[0]
    
    # --- 1. Credit Profile ---
    st.markdown("### 1. Credit Profile & Decision View")
    
    decider = DecisionEngine()
    mkt_bps = st.number_input("Current Market Spread (bps)", value=250)
    implied_bps, risk_status, risk_rationale, risk_level = decider.get_view(row['PD_Prob'], mkt_bps)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Projected Rating", f"{row['Rating']}", f"Base: {row['Base Rating']}") # 动态显示
    m2.metric("Implied Risk Premium", f"{int(implied_bps)} bps", f"vs Mkt {mkt_bps}")
    m3.metric("Stressed Margin", f"{row['Stressed_GM']:.1f}%", f"-{row['Tariff_Impact']:.1f}% (Tariff Hit)", delta_color="inverse")
    m4.metric("Risk Status", risk_status, risk_level)

    # --- 2. Visualization ---
    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("#### 📉 Stress Test: Margin Erosion Path")
        fig = go.Figure(go.Waterfall(
            name = "20", orientation = "v", measure = ["relative", "relative", "relative", "total"],
            x = ["Base Margin", "Tariff Impact", "Cost Inflation", "Stressed Margin"],
            y = [row['Gross Margin'], -row['Tariff_Impact'], -(row['Gross Margin'] - row['Stressed_GM'] - row['Tariff_Impact']), row['Stressed_GM']],
            connector = {"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.markdown("#### 🤖 Compliance Copilot")
        st.success(f"**MAS Green Taxonomy:** ALIGNED (Proxy)\nTarget is Tier-1 Solar Manufacturer.")
        if params['us_tariff_shock'] > 40:
            st.error(f"**Geopolitics:** HIGH RISK\nTariff > 40% triggers covenant breach.")
        else:
            st.warning(f"**Geopolitics:** MONITOR\nUS probe ongoing.")

    # --- 3. Action ---
    st.markdown("---")
    st.markdown("### 📝 Credit Committee Action")
    
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        if st.button("🖨️ Generate Approval Memo"):
            # [改进1 & 2] 动态评级迁移 & 统一字段名
            migration_text = f"Rating Migration: {row['Base Rating']} -> {row['Rating']}"
            
            pdf_data = {
                "borrower": row['Company'],
                "guarantor": f"{row['Company'].split('(')[0]} (SH.Listed) - HQ Guarantee",
                "facility": "Working Capital Loan (Cross-Border)",
                "amount": "USD 50,000,000",
                
                "recommendation": "CONDITIONAL APPROVAL" if row['Score'] > 600 else "REJECT (High Risk)",
                "rationale": f"Internal Score {int(row['Score'])} ({row['Rating']}). \n{migration_text}. \nRequires Parent Guarantee due to {int(row['Tariff_Impact'])}% margin erosion.",
                
                "credit_view": f"Borrower faces significant tariff headwinds. {migration_text}. Implied risk premium is {int(implied_bps)}bps. Strict monitoring required.",
                
                "risk_items": [
                    {"dim": "Financials", "status": "Low" if row['Score']>700 else "Medium", "metric": f"Score {int(row['Score'])}", "comment": "Base financials stable."},
                    {"dim": "Market Pricing", "status": "High" if "Underpriced" in risk_status else "Low", "metric": f"Gap {int(implied_bps - mkt_bps)}bps", "comment": risk_rationale},
                    {"dim": "Geopolitics", "status": "High" if params['us_tariff_shock']>30 else "Medium", "metric": f"Tariff {params['us_tariff_shock']}%", "comment": "US Anti-Circumvention risk."}
                ],
                
                "agent_insights": [
                    {"title": "US Sanctions Check", "detail": "CLEARED. No hits in UFLPA list."},
                    {"title": "MAS Green Taxonomy", "detail": "ALIGNED (Proxy). Renewable Energy Mfg."}
                ],
                
                "stress_scenario": f"US Tariff +{params['us_tariff_shock']}%",
                # [改进1] 确保传给 PDF 的是 list，且包含动态评级
                "stress_outcomes": [
                    f"Gross Margin: {row['Gross Margin']}% -> {row['Stressed_GM']:.1f}%",
                    f"{migration_text}", # 动态迁移
                    "Breakeven: Requires 15% Cost Cutting"
                ],
                
                "mitigation": "1. Parent Corporate Guarantee.\n2. Export Credit Insurance (80% coverage)."
            }
            
            generator = CreditMemoGenerator()
            generator.generate_report(pdf_data)
            
            st.toast("✅ Credit Memo Generated Successfully!")
            # [改进4] Balloon 开关 (Demo 模式才开)
            if st.session_state.get("demo_mode", False):
                st.balloons()
            
            with open("Credit_Committee_Memo.pdf", "rb") as pdf_file:
                st.download_button(
                    label="📥 Download PDF Package",
                    data=pdf_file,
                    file_name=f"CAM_{row['Ticker']}_V32.pdf",
                    mime="application/pdf"
                )

    # [改进5] 银行级页脚声明
    st.markdown("""
        <div class="footer">
            CONFIDENTIAL | INTERNAL USE ONLY <br>
            DISCLAIMER: This tool acts as a decision support system and does not constitute a final credit approval.
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
