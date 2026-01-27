import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import uuid
from datetime import datetime
from fpdf import FPDF
import math

# ==========================================
# 1. 宏观周期模型 (Macro Cycle Model)
# ==========================================
class MacroModel:
    @staticmethod
    def get_cycle_status(year_float):
        """
        模拟光伏行业周期 (正弦波 + 趋势项)
        返回: 周期分数 (-1.0 到 1.0) 和 状态描述
        """
        # 周期逻辑: 约 3-4 年一个短周期
        cycle_component = np.sin(year_float * (2 * np.pi / 3.5)) 
        trend_component = 0.05 * (year_float - 2020) # 长期向上
        macro_score = cycle_component + trend_component
        
        # 状态判定
        if macro_score > 0.5: status = "Overheated (Top)"
        elif macro_score > 0: status = "Expansion (Mid-Cycle)"
        elif macro_score > -0.5: status = "Contraction (Downturn)"
        else: status = "Trough (Bottom)"
        
        return macro_score, status

    @staticmethod
    def plot_cycle_curve():
        years = np.linspace(2020, 2027, 100)
        scores = [MacroModel.get_cycle_status(y)[0] for y in years]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=scores, mode='lines', name='Industry Cycle', line=dict(color='#00E5FF', width=3)))
        
        # 标记当前时间点
        current_year = datetime.now().year + datetime.now().month / 12.0
        current_score, current_status = MacroModel.get_cycle_status(current_year)
        
        fig.add_trace(go.Scatter(x=[current_year], y=[current_score], mode='markers', name='NOW', 
                                marker=dict(size=12, color='#FF3D00', symbol='diamond')))
        
        fig.update_layout(
            title="PV INDUSTRY MACRO CYCLE (THEORY)",
            template="plotly_dark",
            xaxis_title="Year",
            yaxis_title="Cycle Sentiment",
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig, current_status

# ==========================================
# 2. 情景管理器 (Scenario Manager)
# ==========================================
class ScenarioManager:
    SCENARIOS = {
        "Base Case": {
            "margin_shock_bps": 0, "tariff_shock_pct": 0.0, "market_demand_adj": 1.0, "desc": "Current Market Conditions"
        },
        "Trade War 2025 (Severe)": {
            "margin_shock_bps": 300, "tariff_shock_pct": 0.25, "market_demand_adj": 0.8, "desc": "High Tariffs & Export Ban"
        },
        "Price War (Deflation)": {
            "margin_shock_bps": 800, "tariff_shock_pct": 0.0, "market_demand_adj": 1.2, "desc": "Domestic Price War to Clear Inventory"
        },
        "Tech Disruption (N-Type)": {
            "margin_shock_bps": 200, "tariff_shock_pct": 0.05, "market_demand_adj": 0.9, "desc": "Old Capacity Obsolescence"
        }
    }

# ==========================================
# 3. 核心算法: Logistic Regression Logic
# ==========================================
class CreditEnginePro:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def calculate_pd_score(row, scenario_params, macro_status):
        """
        使用 Logit 变换计算违约概率 (PD) 并映射为分数
        Formula: Log(Odds) = Intercept + B1*X1 + B2*X2 ...
        """
        # 1. 因子提取与压力测试
        # 毛利 (Gross Margin)
        base_gm = row['Gross Margin']
        stressed_gm = base_gm - (scenario_params['margin_shock_bps'] / 100.0)
        # 关税冲击 (Tariff Hit)
        overseas_exposure = row['Overseas Ratio'] / 100.0
        tariff_hit = overseas_exposure * scenario_params['tariff_shock_pct'] * 100
        final_gm = stressed_gm - tariff_hit
        
        # 存货周转 (Inventory)
        inv_days = row['Inventory Days']
        
        # 现金流覆盖 (Cash Flow Coverage) -> 简化为 0/1 因子
        cf_flag = 1 if row['Cash Flow'] > 0 else 0
        
        # 2. 宏观校准 (Macro Calibration)
        macro_adj = 0
        if "Downturn" in macro_status or "Trough" in macro_status:
            macro_adj = -0.5 # 宏观环境差，Log-odds 偏向违约
        
        # 3. Logit 模型计算 (模拟系数 - Expert Calibrated)
        # Logit(Default) = Intercept - a*Margin + b*Inventory - c*CashFlow + Macro
        # 注意：这里我们算的是“违约的 Log-odds”，所以因子符号要小心
        # Margin 越高，违约越低 (负号)
        # Inventory 越高，违约越高 (正号)
        
        intercept = -1.0 
        coef_gm = -0.15      # 毛利每高 1%，Log-odds 降低 0.15
        coef_inv = 0.02      # 存货每高 1天，Log-odds 增加 0.02
        coef_cf = -0.8       # 正现金流，Log-odds 降低 0.8
        
        logit_z = intercept + (coef_gm * final_gm) + (coef_inv * inv_days) + (coef_cf * cf_flag) + macro_adj
        
        # 4. 转化为 PD (Probability of Default)
        pd_value = CreditEnginePro.sigmoid(logit_z)
        
        # 5. PD 映射为分数 (Score = 100 * (1 - PD))
        # 做一些平滑处理，避免 0 或 100
        score = 100 * (1 - pd_value)
        
        # 评级
        if score >= 85: rating = "AAA"
        elif score >= 70: rating = "AA"
        elif score >= 55: rating = "BBB"
        elif score >= 40: rating = "BB"
        elif score >= 25: rating = "B"
        else: rating = "CCC"
        
        return pd.Series({
            'Stressed_GM': final_gm,
            'PD_Prob': pd_value,
            'V18_Score': score,
            'Rating': rating,
            'Logit_Z': logit_z
        })

# ==========================================
# 4. UI 渲染引擎 (V18.0 Black Gold)
# ==========================================
st.set_page_config(page_title="Global Credit Lens V18.0", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    .stApp { background-color: #000000 !important; color: #E0E0E0; font-family: 'Helvetica Neue', sans-serif; }
    [data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #333; }
    h1, h2, h3 { color: #00E5FF !important; text-transform: uppercase; font-weight: 800 !important; }
    .stMetric { background-color: #111; border: 1px solid #333; border-left: 4px solid #0056D2; padding: 15px; }
    .stSelectbox label { color: #00E5FF !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 5. 主程序逻辑
# ==========================================
def main():
    # --- 侧边栏：情景与参数 ---
    st.sidebar.title("🎮 SCENARIO LAB")
    
    # 1. 宏观周期展示
    st.sidebar.markdown("### 1. MACRO CYCLE POSITION")
    macro_fig, macro_status = MacroModel.plot_cycle_curve()
    st.sidebar.plotly_chart(macro_fig, use_container_width=True)
    st.sidebar.info(f"Current Phase: **{macro_status}**")
    
    # 2. 情景选择器
    st.sidebar.markdown("### 2. STRESS SCENARIO")
    selected_scenario_name = st.sidebar.selectbox("Select Market Scenario", list(ScenarioManager.SCENARIOS.keys()))
    scenario_params = ScenarioManager.SCENARIOS[selected_scenario_name]
    
    # 展示参数详情
    with st.sidebar.expander("Scenario Parameters", expanded=True):
        st.write(f"📉 Margin Compression: **{scenario_params['margin_shock_bps']} bps**")
        st.write(f"🚢 Tariff Shock: **{scenario_params['tariff_shock_pct']:.0%}**")
        st.write(f"🛒 Demand Adj: **{scenario_params['market_demand_adj']}x**")
        st.caption(f"📝 {scenario_params['desc']}")

    # --- 主界面 ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("GLOBAL CREDIT LENS | V18.0")
        st.caption(f"LOGIT-BASED PROBABILITY OF DEFAULT MODEL | MODE: {selected_scenario_name.upper()}")
    with c2:
        st.metric("MODEL ENGINE", "LOGISTIC REGRESSION", "Sigmoid Activation")

    # --- 数据模拟 (Mock Data) ---
    data = [
        {'Ticker': '600438.SH', 'Company Name': 'Tongwei Solar', 'Gross Margin': 28.5, 'Overseas Ratio': 25.0, 'Inventory Days': 85, 'Cash Flow': 1},
        {'Ticker': '300750.SZ', 'Company Name': 'CATL', 'Gross Margin': 22.0, 'Overseas Ratio': 35.0, 'Inventory Days': 70, 'Cash Flow': 1},
        {'Ticker': '688599.SH', 'Company Name': 'Trina Solar', 'Gross Margin': 15.5, 'Overseas Ratio': 60.0, 'Inventory Days': 110, 'Cash Flow': 0},
        {'Ticker': '002459.SZ', 'Company Name': 'Jinko Power', 'Gross Margin': 14.0, 'Overseas Ratio': 72.0, 'Inventory Days': 140, 'Cash Flow': 1},
        {'Ticker': '601012.SH', 'Company Name': 'Longi Green', 'Gross Margin': 18.0, 'Overseas Ratio': 45.0, 'Inventory Days': 95, 'Cash Flow': 1}
    ]
    df = pd.DataFrame(data)

    # --- 计算引擎运行 ---
    # 传入宏观状态和情景参数
    res = df.apply(lambda r: CreditEnginePro.calculate_pd_score(r, scenario_params, macro_status), axis=1)
    df_final = pd.concat([df, res], axis=1)

    # --- 结果可视化 ---
    
    # Row 1: 核心指标
    k1, k2, k3, k4 = st.columns(4)
    avg_pd = df_final['PD_Prob'].mean()
    high_risk_num = len(df_final[df_final['Rating'].isin(['B', 'CCC'])])
    
    k1.metric("PORTFOLIO AVG PD", f"{avg_pd:.2%}", delta="Probability of Default", delta_color="inverse")
    k2.metric("AVG CREDIT SCORE", f"{df_final['V18_Score'].mean():.1f}", delta="Logit Mapped")
    k3.metric("HIGH RISK ENTITIES", str(high_risk_num), delta="Watch List", delta_color="inverse")
    k4.metric("MACRO OVERLAY", macro_status, "Cycle Adjustment Applied")

    # Row 2: PD 曲线与气泡图
    st.markdown("### 📊 RISK TOPOGRAPHY")
    t1, t2 = st.columns([2, 1])
    
    with t1:
        # 绘制 S 曲线 (Sigmoid Curve) 展示位置
        x_range = np.linspace(-6, 6, 100)
        y_range = 1 / (1 + np.exp(-x_range))
        
        fig_logit = go.Figure()
        fig_logit.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='Logistic Function', line=dict(color='#444', dash='dash')))
        
        # 将公司投射到 S 曲线上
        fig_logit.add_trace(go.Scatter(
            x=df_final['Logit_Z'], 
            y=df_final['PD_Prob'], 
            mode='markers+text',
            text=df_final['Company Name'],
            textposition='top center',
            marker=dict(size=12, color=df_final['V18_Score'], colorscale='RdYlGn', showscale=True),
            name='Companies'
        ))
        
        fig_logit.update_layout(
            title="Logistic Mapping (Z-Score to PD)",
            xaxis_title="Logit Z-Score (Higher = More Risk)",
            yaxis_title="Probability of Default (PD)",
            template="plotly_dark",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_logit, use_container_width=True)
        st.caption("Mathematical Core: $PD = 1 / (1 + e^{-z})$, where $z = \\alpha + \\beta_1 X_1 + ... + Macro$")

    with t2:
        st.markdown("#### SCENARIO IMPACT")
        st.dataframe(
            df_final[['Company Name', 'Rating', 'PD_Prob', 'Stressed_GM']]
            .style.format({'PD_Prob': "{:.2%}", 'Stressed_GM': "{:.1f}%"})
            .background_gradient(subset=['PD_Prob'], cmap='Reds'),
            use_container_width=True,
            height=400
        )

    # --- PDF 生成模块 (含宏观与情景描述) ---
    st.markdown("### 📑 AUDITED REPORTING")
    
    col_pdf_sel, col_pdf_btn = st.columns([3, 1])
    target_comp = col_pdf_sel.selectbox("Select Issuer for Memo", df_final['Company Name'])
    
    if col_pdf_btn.button("GENERATE V18 REPORT"):
        row = df_final[df_final['Company Name'] == target_comp].iloc[0]
        
        # --- PDF 生成逻辑 ---
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"CREDIT RISK MEMO: {target_comp}", 0, 1)
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 10, f"Report ID: {str(uuid.uuid4())[:8].upper()} | Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
        pdf.line(10, 30, 200, 30)
        
        # Macro Section
        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "1. MACRO & SCENARIO CONTEXT", 0, 1)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 6, f"Cycle Status: {macro_status}\nScenario Applied: {selected_scenario_name}\nDescription: {scenario_params['desc']}")
        
        # Financial Impact
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "2. STRESS TEST RESULTS (LOGIT MODEL)", 0, 1)
        pdf.set_font("Courier", "", 10)
        
        pdf.cell(100, 8, f"Original Margin   : {row['Gross Margin']:.2f}%", 0, 1)
        pdf.cell(100, 8, f"Stressed Margin   : {row['Stressed_GM']:.2f}%", 0, 1)
        pdf.cell(100, 8, f"Logit Z-Score     : {row['Logit_Z']:.4f}", 0, 1)
        pdf.cell(100, 8, f"Prob. of Default  : {row['PD_Prob']:.2%}", 0, 1)
        pdf.cell(100, 8, f"Implied Rating    : {row['Rating']}", 0, 1)
        
        # Disclaimer
        pdf.set_y(-30)
        pdf.set_font("Arial", "I", 8)
        pdf.multi_cell(0, 5, "Model Methodology: Logistic Regression based on expert-calibrated coefficients. PD represents 12-month forward-looking probability under stressed assumptions.")
        
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button("📥 DOWNLOAD PDF", pdf_bytes, "V18_FullStack_Report.pdf", "application/pdf")

if __name__ == "__main__":
    main()
