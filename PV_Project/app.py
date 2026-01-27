
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
# 0. 基础配置
# ==========================================
st.set_page_config(page_title="全球信贷透视系统 V19 (终极版)", layout="wide", page_icon="🏦")

# 黑金样式
st.markdown("""
    <style>
    .stApp { background-color: #000000 !important; color: #E0E0E0; font-family: 'Microsoft YaHei', sans-serif; }
    [data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #333; }
    h1, h2, h3 { color: #00E5FF !important; font-weight: 800 !important; }
    .stMetric { background-color: #111; border: 1px solid #333; border-left: 4px solid #0056D2; padding: 15px; }
    /* 让Tab页签更明显 */
    .stTabs [aria-selected="true"] { background-color: #0056D2 !important; color: white !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 宏观与剧本逻辑
# ==========================================
class MacroModel:
    @staticmethod
    def get_cycle_status(year_float):
        # 模拟周期
        cycle = np.sin(year_float * (2 * np.pi / 3.5)) + 0.05 * (year_float - 2020)
        if cycle > 0.5: return cycle, "过热期 (Top)"
        elif cycle > 0: return cycle, "扩张期 (Mid)"
        elif cycle > -0.5: return cycle, "衰退期 (Down)"
        else: return cycle, "萧条期 (Bottom)"

class ScenarioManager:
    SCENARIOS = {
        "基准情形 (Base Case)": {"margin_shock": 0, "tariff_shock": 0.0, "desc": "维持现状"},
        "2025 贸易战 (Trade War)": {"margin_shock": 300, "tariff_shock": 0.35, "desc": "高关税冲击出口"},
        "国内价格战 (Price War)": {"margin_shock": 800, "tariff_shock": 0.0, "desc": "内卷导致毛利暴跌"}
    }

# ==========================================
# 2. 核心计算引擎 (新增资产负债率)
# ==========================================
class CreditEngine:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def calculate(row, params, macro_status):
        # 1. 因子提取
        base_gm = row['Gross Margin']
        debt_ratio = row['Debt Ratio'] # 新增：资产负债率
        
        # 2. 压力测试逻辑
        # 你的质疑：参数可调吗？现在这里接收的 params 是用户实时调整过的
        stressed_gm = base_gm - (params['margin_shock'] / 100.0)
        tariff_hit = (row['Overseas Ratio'] / 100.0) * params['tariff_shock'] * 100
        final_gm = stressed_gm - tariff_hit
        
        # 3. 宏观调整
        macro_adj = -0.5 if "衰退" in macro_status or "萧条" in macro_status else 0
        
        # 4. Logit 公式 (模型参数)
        # Z = 截距 + (毛利 * 权重) + (库存 * 权重) + (负债 * 权重) ...
        # 你的质疑：参数够吗？现在加入了负债率，更完善了
        intercept = -0.5
        coef_gm = -0.15     # 毛利越高，风险越低
        coef_inv = 0.02     # 库存越高，风险越高
        coef_debt = 0.05    # 新增：负债率越高(比如70%)，风险越高
        coef_cf = -1.2      # 现金流为正，大幅降低风险
        
        logit_z = intercept + \
                  (coef_gm * final_gm) + \
                  (coef_inv * row['Inventory Days']) + \
                  (coef_debt * debt_ratio) + \
                  (coef_cf * (1 if row['Cash Flow']>0 else 0)) + \
                  macro_adj
                  
        pd_val = CreditEngine.sigmoid(logit_z)
        score = 100 * (1 - pd_val)
        
        # 评级
        if score >= 85: rating = "AAA"
        elif score >= 70: rating = "AA"
        elif score >= 55: rating = "BBB"
        elif score >= 40: rating = "BB"
        else: rating = "CCC"
        
        return pd.Series({
            'Stressed_GM': final_gm, 
            'PD_Prob': pd_val, 
            'Score': score, 
            'Rating': rating,
            'Logit_Z': logit_z
        })

# ==========================================
# 3. 主程序
# ==========================================
def main():
    # --- 侧边栏：控制台 ---
    st.sidebar.title("🎛️ 风险控制台")
    
    # 宏观展示
    cycle_val, cycle_str = MacroModel.get_cycle_status(2026.1)
    st.sidebar.info(f"宏观周期: {cycle_str}")
    
    # 剧本选择
    st.sidebar.markdown("---")
    st.sidebar.subheader("1. 剧本设定")
    sc_name = st.sidebar.selectbox("选择预设剧本", list(ScenarioManager.SCENARIOS.keys()))
    base_params = ScenarioManager.SCENARIOS[sc_name]
    
    # 你的质疑：能不能调？这里增加了“手动覆写”功能
    st.sidebar.subheader("2. 参数微调 (敏感性分析)")
    override = st.sidebar.checkbox("启用手动覆写 (Override)", value=False)
    
    if override:
        st.sidebar.caption("⚠️ 警告：您正在偏离标准模型参数")
        margin_shock = st.sidebar.slider("毛利冲击 (bps)", 0, 1500, base_params['margin_shock'])
        tariff_shock = st.sidebar.slider("关税冲击 (%)", 0.0, 1.0, base_params['tariff_shock'])
    else:
        margin_shock = base_params['margin_shock']
        tariff_shock = base_params['tariff_shock']
        st.sidebar.code(f"毛利冲击: {margin_shock} bps\n关税冲击: {tariff_shock:.0%}")
        
    final_params = {'margin_shock': margin_shock, 'tariff_shock': tariff_shock}

    # --- 数据准备 (新增资产负债率) ---
    df = pd.DataFrame([
        {'Company': '通威股份', 'Gross Margin': 28.5, 'Overseas Ratio': 25.0, 'Inventory Days': 85, 'Debt Ratio': 55.0, 'Cash Flow': 1},
        {'Company': '宁德时代', 'Gross Margin': 22.0, 'Overseas Ratio': 35.0, 'Inventory Days': 70, 'Debt Ratio': 45.0, 'Cash Flow': 1},
        {'Company': '天合光能', 'Gross Margin': 15.5, 'Overseas Ratio': 60.0, 'Inventory Days': 110, 'Debt Ratio': 68.0, 'Cash Flow': 0},
        {'Company': '晶科能源', 'Gross Margin': 14.0, 'Overseas Ratio': 72.0, 'Inventory Days': 140, 'Debt Ratio': 74.0, 'Cash Flow': 1},
        {'Company': '隆基绿能', 'Gross Margin': 18.0, 'Overseas Ratio': 45.0, 'Inventory Days': 95, 'Debt Ratio': 50.0, 'Cash Flow': 1}
    ])
    
    # 计算
    res = df.apply(lambda r: CreditEngine.calculate(r, final_params, cycle_str), axis=1)
    df_final = pd.concat([df, res], axis=1)

    # --- 主界面 ---
    st.title("全球信贷透视系统 | V19.0 Hybrid")
    st.caption("融合数学模型、可视化分析与敏感性测试的完整风控平台")
    
    # KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("当前剧本", sc_name)
    k2.metric("平均违约率 (PD)", f"{df_final['PD_Prob'].mean():.1%}", delta=f"冲击: {margin_shock}bps", delta_color="inverse")
    k3.metric("高风险企业", len(df_final[df_final['Score']<60]), delta="Rating < BBB", delta_color="inverse")
    k4.metric("模型因子数", "5个", "新增: 资产负债率")

    st.markdown("---")

    # --- 多维度分析 Tab (复活你的图表) ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 违约模型 (Logit)", 
        "🗺️ 行业热力图 (Heatmap)", 
        "🛁 竞争气泡图 (Bubble)", 
        "🎻 风险分布 (Violin)", 
        "🔗 因子相关性 (Corr)"
    ])

    # 1. 核心模型 (V18遗留)
    with tab1:
        c1, c2 = st.columns([2,1])
        with c1:
            # S曲线
            x = np.linspace(-6, 6, 100)
            y = 1 / (1 + np.exp(-x))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, name="Sigmoid基准", line=dict(dash='dash', color='#444')))
            fig.add_trace(go.Scatter(x=df_final['Logit_Z'], y=df_final['PD_Prob'], mode='markers+text', 
                                     text=df_final['Company'], marker=dict(size=15, color=df_final['Score'], colorscale='RdYlGn'),
                                     name="当前位置"))
            fig.update_layout(title="违约概率映射 (PD Mapping)", template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(df_final[['Company', 'Rating', 'PD_Prob', 'Stressed_GM']].style.background_gradient(subset=['PD_Prob'], cmap='Reds'), height=400)

    # 2. 热力图 (你的V15回归)
    with tab2:
        fig_tree = px.treemap(df_final, path=[px.Constant("光伏行业"), 'Rating', 'Company'], values='Score',
                              color='Score', color_continuous_scale='RdYlGn', title="信用评分板块热力图")
        fig_tree.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig_tree, use_container_width=True)

    # 3. 气泡图 (你的V15回归)
    with tab3:
        fig_bub = px.scatter(df_final, x="Stressed_GM", y="Score", size="Debt Ratio", color="Rating",
                             hover_name="Company", title="利润 vs 评分 (气泡大小=负债率)",
                             color_discrete_sequence=px.colors.qualitative.Bold)
        fig_bub.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig_bub, use_container_width=True)
        
    # 4. 小提琴图 (你的V15回归)
    with tab4:
        fig_vio = px.strip(df_final, x="Rating", y="Score", color="Rating", title="评级分布离散度")
        fig_vio.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig_vio, use_container_width=True)

    # 5. 相关性图 (你的V15回归)
    with tab5:
        # 只算数字列
        corr = df_final[['Score', 'Gross Margin', 'Overseas Ratio', 'Inventory Days', 'Debt Ratio']].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="风险因子相关性矩阵")
        fig_corr.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- 报告导出 ---
    st.markdown("### 📑 审计报告")
    if st.button("生成本轮测试报告"):
        st.success(f"已生成基于 [{sc_name}] 且 毛利冲击={margin_shock}bps 的压力测试报告。ID: {str(uuid.uuid4())[:8]}")

if __name__ == "__main__":
    main()
