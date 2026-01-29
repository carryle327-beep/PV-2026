import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from fpdf import FPDF
import io

# ==========================================
# 0. 系统配置 (V26.5 校准修复版)
# ==========================================
st.set_page_config(page_title="Global Credit Lens V26.5", layout="wide", page_icon="🏦")

# CSS 样式: 黑金/投行风 (优化字体清晰度)
st.markdown("""
    <style>
    .stApp { background-color: #000000 !important; color: #E0E0E0; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background-color: #121212 !important; border-right: 1px solid #333; }
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 600 !important; }
    .stMetric { background-color: #1A1A1A; border: 1px solid #333; border-left: 4px solid #0056D2; padding: 15px; border-radius: 5px; }
    .stButton>button { background-color: #222; color: white; border: 1px solid #444; }
    .streamlit-expanderHeader { background-color: #222 !important; color: white !important; }
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
# 2. 核心计算引擎 (Logit + PDO Scaling)
# ==========================================
class CreditEngine:
    @staticmethod
    def sigmoid(z): return 1 / (1 + np.exp(-z))

    @staticmethod
    def scale_score(pd_val, base_score=600, base_odds=20, pdo=40):
        """
        [校准修复] 
        1. base_odds 从 50 降为 20 (降低达到600分的门槛)
        2. pdo 从 20 升为 40 (拉大分差，让高分更高，低分更低，增加区分度)
        """
        if pd_val >= 1.0: return 300
        if pd_val <= 0.0: return 850
        
        factor = pdo / np.log(2)
        offset = base_score - (factor * np.log(base_odds))
        current_odds = (1 - pd_val) / pd_val
        score = offset + (factor * np.log(current_odds))
        return int(max(300, min(850, score)))

    @staticmethod
    def calculate(row, params, macro_status):
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

        # [校准修复] Logit 回归参数调整
        # Intercept 从 -0.5 调整为 -2.0。这意味着所有人天生自带“安全分”，降低基础 PD。
        logit_z = -2.0 + (-0.12 * final_gm) + (0.015 * inv) + (0.04 * debt_ratio) + (-1.5 * cf_flag) + rate_hit
        
        pd_val = CreditEngine.sigmoid(logit_z)
        
        # PDO 分数校准
        score = CreditEngine.scale_score(pd_val, base_score=600, base_odds=20, pdo=40)
        
        # 评级映射
        if score >= 750: rating = "AA (High Grade)"
        elif score >= 700: rating = "A (Upper Medium)"
        elif score >= 650: rating = "BBB (Medium)"
        elif score >= 580: rating = "BB (Speculative)" # 稍微降低 BB 门槛
        elif score >= 500: rating = "B (Highly Speculative)"
        else: rating = "CCC (Substantial Risk)"
        
        return pd.Series({'Stressed_GM': final_gm, 'PD_Prob': pd_val, 'Score': score, 'Rating': rating})

# ==========================================
# 3. 巴塞尔资本引擎
# ==========================================
class BaselEngine:
    def __init__(self):
        self.rw_map = {'AA (High Grade)': 0.20, 'A (Upper Medium)': 0.50, 'BBB (Medium)': 1.00, 
                       'BB (Speculative)': 1.00, 'B (Highly Speculative)': 1.50, 'CCC (Substantial Risk)': 1.50}
        self.capital_ratio = 0.08 

    def calculate_rwa(self, exposure, rating):
        rw = 1.50
        for key in self.rw_map:
            if rating.startswith(key.split(' ')[0]):
                rw = self.rw_map[key]
                break
        rwa = exposure * rw
        charge = rwa * self.capital_ratio
        return rw, rwa, charge

# ==========================================
# 4. 黑天鹅引擎
# ==========================================
class BlackSwanEngine:
    @staticmethod
    def simulate_survival(row, shock_factor, fixed_cost_ratio=0.25):
        gross_margin = float(row.get('Gross Margin', 20)) / 100.0
        base_revenue = 100.0
        base_cogs = base_revenue * (1 - gross_margin)
        base_fixed_cost = base_revenue * fixed_cost_ratio 
        base_net_profit = base_revenue - base_cogs - base_fixed_cost
        
        new_revenue = base_revenue * (1 - shock_factor)
        new_cogs = new_revenue * (1 - gross_margin)
        new_net_profit = new_revenue - new_cogs - base_fixed_cost
        
        return {
            'Base_Profit': base_net_profit,
            'Impact': new_net_profit - base_net_profit,
            'Final_Profit': new_net_profit,
            'Is_Survive': new_net_profit > 0
        }

# ==========================================
# 5. IV 计算引擎
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
# 6. 主程序
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

    # B. 宏观压力参数
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. 宏观压力参数")
    params = {
        'margin_shock': st.sidebar.slider("1. 行业内卷 (bps)", 0, 1000, 300),
        'tariff_shock': st.sidebar.slider("2. 关税壁垒 (%)", 0.0, 1.0, 0.25),
        'rate_hike_bps': st.sidebar.slider("3. 美联储加息 (bps)", 0, 500, 100),
        'raw_material_shock': st.sidebar.slider("4. 原材料通胀 (%)", 0, 50, 10),
        'fx_shock': st.sidebar.slider("5. 汇率风险 (%)", 0, 20, 5)
    }
    
    # C. 黑天鹅参数
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏴‍☠️ 黑天鹅生成器")
    st.sidebar.caption("Tail Risk & Operating Leverage Simulation")
    swan_shock = st.sidebar.slider("📉 营收瞬间蒸发 (%)", 0, 80, 40)
    fixed_cost_sim = st.sidebar.slider("🏭 假设固定成本占比 (%)", 10, 60, 25)

    # D. 计算
    try:
        res_stressed = df_raw.apply(lambda r: CreditEngine.calculate(r, params, "Stressed"), axis=1)
        df_final = pd.concat([df_raw, res_stressed], axis=1)
        base_params = {k:0 for k in params}
        res_base = df_raw.apply(lambda r: CreditEngine.calculate(r, base_params, "Base"), axis=1)
        df_final['Base_Rating'] = res_base['Rating']
        df_final['Search_Label'] = df_final['Ticker'] + " | " + df_final['Company']
    except: return

    # ==========================================
    # 前端展示层
    # ==========================================
    st.title("GLOBAL CREDIT LENS | V26.5")
    st.caption("Architect Edition: Calibrated Scoring + High Contrast Viz")
    
    # 检索
    c_search, _ = st.columns([1, 2])
    with c_search:
        selected_label = st.selectbox("🔍 穿透式检索 (Drill-down)", df_final['Search_Label'].tolist())
    
    selected_ticker = selected_label.split(" | ")[0]
    row = df_final[df_final['Ticker'] == selected_ticker].iloc[0]

    # --- 第一排：信用仪表盘 & RWA 资本瀑布 ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🧬 Credit Profile (PDO Scaled)")
        
        # 仪表盘颜色逻辑：根据分数变色
        rating_color = '#28A745' if row['Score'] >= 650 else ('#FFC107' if row['Score'] >= 580 else '#DC3545')
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta", value = row['Score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            delta = {'reference': 600, 'increasing': {'color': "#28A745"}},
            gauge = {
                'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': rating_color},
                'bgcolor': "black", 'borderwidth': 2, 'bordercolor': "#333",
                'steps': [
                    {'range': [300, 580], 'color': '#550000'}, # 红
                    {'range': [580, 650], 'color': '#555500'}, # 黄
                    {'range': [650, 850], 'color': '#003300'}  # 绿
                ],
                'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': row['Score']}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.metric("Current Rating", row['Rating'], delta=f"PD: {row['PD_Prob']:.2%}", delta_color="inverse")

    with col2:
        st.markdown("### 🏛️ Basel III Capital Impact")
        exposure = 10_000_000 
        basel = BaselEngine()
        rw_base, rwa_base, cap_base = basel.calculate_rwa(exposure, row['Base_Rating'])
        rw_stress, rwa_stress, cap_stress = basel.calculate_rwa(exposure, row['Rating'])
        cap_delta = cap_stress - cap_base
        
        # [修复] 瀑布图文字可见性：强制白色 + 加粗
        fig_cap = go.Figure(go.Waterfall(
            name = "Capital", orientation = "v",
            measure = ["relative", "relative", "total"],
            x = ["Base Capital", "Stress Impact", "Final Required"],
            textposition = "auto", # 改为 auto，防止 outside 被切掉
            text = [f"${cap_base/1000:.0f}k", f"+${cap_delta/1000:.0f}k", f"${cap_stress/1000:.0f}k"],
            textfont = dict(color="white", size=14, family="Arial Black"), # 强制白色大字体
            y = [cap_base, cap_delta, cap_stress],
            connector = {"line":{"color":"#666"}},
            increasing = {"marker":{"color":"#FF4B4B"}}, decreasing = {"marker":{"color":"#28A745"}}, totals = {"marker":{"color":"#EEE"}}
        ))
        # 增加 cliponaxis=False 防止文字被切
        fig_cap.update_layout(title="Capital Erosion (USD)", template="plotly_dark", height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig_cap.update_yaxes(automargin=True)
        st.plotly_chart(fig_cap, use_container_width=True)

    # --- 第二排：黑天鹅生存模拟 ---
    st.markdown("---")
    st.subheader("🏴‍☠️ Black Swan Survival Test")
    
    swan_res = BlackSwanEngine.simulate_survival(row, swan_shock/100.0, fixed_cost_sim/100.0)
    c_swan1, c_swan2 = st.columns([1, 2])
    
    with c_swan1:
        is_alive = swan_res['Is_Survive']
        status_color = '#28A745' if is_alive else '#FF3333'
        status_text = "SURVIVED" if is_alive else "BANKRUPT"
        st.markdown(f"""
            <div style="background-color:#111; padding:20px; border: 1px solid {status_color}; border-radius: 8px; text-align:center; height: 100%;">
                <h4 style="color:#888; margin:0;">SIMULATION RESULT</h4>
                <h1 style="color:{status_color}; font-size:48px; margin:15px 0;">{status_text}</h1>
                <p style="color:#AAA;">Revenue Shock: <b style="color:#FFF">-{swan_shock}%</b></p>
                <p style="color:#AAA;">Operating Leverage Impact</p>
            </div>
        """, unsafe_allow_html=True)
        
    with c_swan2:
        # [修复] 瀑布图文字可见性
        fig_swan = go.Figure(go.Waterfall(
            name = "Survival", orientation = "v",
            measure = ["relative", "relative", "total"],
            x = ["Base Profit", "Shock Impact", "Final Profit"],
            textposition = "auto",
            text = [f"{swan_res['Base_Profit']:.1f}", f"{swan_res['Impact']:.1f}", f"{swan_res['Final_Profit']:.1f}"],
            textfont = dict(color="white", size=14, family="Arial Black"), # 强制白色大字体
            y = [swan_res['Base_Profit'], swan_res['Impact'], swan_res['Final_Profit']],
            connector = {"line":{"color":"#666"}},
            increasing = {"marker":{"color":"#28A745"}}, decreasing = {"marker":{"color":"#FF3333"}}, totals = {"marker":{"color": "#FFF" if is_alive else "#555"}}
        ))
        fig_swan.update_layout(title=f"Operating Leverage Analysis (Fixed Cost Ratio: {fixed_cost_sim}%)", template="plotly_dark", height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_swan, use_container_width=True)

    # --- 第三排：量化看板 ---
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Risk Heatmap", "🎻 Rating Dist", "🔗 Correlations", "🧠 IV Feature Selection"])

    with tab1:
        if not df_final.empty:
            st.plotly_chart(px.treemap(df_final, path=[px.Constant("Market"), 'Rating', 'Search_Label'], values='Score', color='Score', color_continuous_scale='RdYlGn', title="Credit Risk Heatmap"), use_container_width=True)
    with tab2:
        if not df_final.empty:
            st.plotly_chart(px.strip(df_final, x="Rating", y="Score", color="Rating", title="Rating Distribution"), use_container_width=True)
    with tab3:
        if not df_final.empty:
            st.plotly_chart(px.imshow(df_final[['Score', 'Gross Margin', 'Overseas Ratio', 'Inventory Days', 'Debt Ratio']].corr(), text_auto=True, color_continuous_scale='RdBu_r', title="Factor Correlation Matrix"), use_container_width=True)
    with tab4:
        st.markdown("#### Information Value (IV) Analysis")
        if not df_final.empty:
            target_col = 'Manual_Bad_Label' if 'Manual_Bad_Label' in df_final.columns else 'Is_Bad'
            if target_col == 'Is_Bad': df_final['Is_Bad'] = df_final['PD_Prob'].apply(lambda x: 1 if x > 0.30 else 0)
            
            iv_result = IV_Engine.calculate_iv(df_final, target_col=target_col, feature_cols=['Gross Margin', 'Debt Ratio', 'Overseas Ratio', 'Inventory Days', 'Cash Flow'])
            iv_result['Color'] = iv_result['IV'].apply(lambda x: '#FFD700' if x > 0.3 else ('#00E5FF' if x > 0.1 else '#555'))
            st.plotly_chart(px.bar(iv_result, x='IV', y='Feature', orientation='h', title="Predictive Power (IV)", color='Feature', color_discrete_map={row['Feature']: row['Color'] for _, row in iv_result.iterrows()}).update_layout(template="plotly_dark", showlegend=False, height=300), use_container_width=True)

    # --- 底部：架构图与报告 ---
    st.markdown("---")
    c_rep, c_arch = st.columns([1, 2])
    
    with c_rep:
        if st.button(f"📄 Generate Investment Memo ({row['Ticker']})"):
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 20)
                pdf.cell(0, 15, f"CREDIT MEMO: {row['Ticker']}", 0, 1)
                pdf.line(10, 25, 200, 25)
                pdf.ln(5)
                pdf.set_font("Arial", "", 12)
                pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
                pdf.cell(0, 10, f"Rating: {row['Rating']} (Score: {row['Score']})", 0, 1)
                pdf.cell(0, 10, f"PD: {row['PD_Prob']:.2%}", 0, 1)
                pdf.ln(5)
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "STRESS TEST & CAPITAL IMPACT", 0, 1)
                pdf.set_font("Arial", "", 12)
                pdf.cell(0, 8, f"Capital Charge Delta: +${cap_delta:,.0f}", 0, 1)
                pdf.cell(0, 8, f"Black Swan Survival: {'YES' if is_alive else 'NO'}", 0, 1)
                pdf.ln(10)
                pdf.set_font("Arial", "I", 8)
                pdf.cell(0, 10, "Generated by Global Credit Lens V26.5", 0, 1)
                st.download_button("📥 Download PDF", bytes(pdf.output()), f"Report_{row['Ticker']}.pdf")
            except: st.error("PDF Generation Error")

    with c_arch:
        with st.expander("🏗️ System Architecture (Text View)", expanded=False):
            st.markdown("""
            **1. Presentation Layer:** Streamlit Dashboard, High-Contrast Waterfall Charts, FPDF.
            **2. Core Computing Layer:** * **Scoring:** Calibrated Logit (Base PD adjustment) + PDO Scaling.
            * **Stress:** 5-Factor Macro Shock Simulation.
            * **Capital:** Basel III Standardized Approach.
            * **Survival:** Operating Leverage & Black Swan Engine.
            **3. Data Layer:** Excel Feed / SQL Adapter.
            """)

if __name__ == "__main__":
    main()
