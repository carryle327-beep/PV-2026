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
# 1. 宏观周期模型 (决定是大环境好还是坏)
# ==========================================
class MacroModel:
    @staticmethod
    def get_cycle_status(year_float):
        """
        用正弦波模拟光伏行业的“看天吃饭”
        返回: 宏观得分, 中文状态描述
        """
        # 模拟 3.5 年一个周期
        cycle_component = np.sin(year_float * (2 * np.pi / 3.5)) 
        trend_component = 0.05 * (year_float - 2020) # 行业长期是向上的
        macro_score = cycle_component + trend_component
        
        # 翻译成中文状态
        if macro_score > 0.5: status = "过热期 (顶部风险)"
        elif macro_score > 0: status = "扩张期 (复苏中)"
        elif macro_score > -0.5: status = "衰退期 (下行压力)"
        else: status = "萧条期 (谷底磨底)"
        
        return macro_score, status

    @staticmethod
    def plot_cycle_curve():
        years = np.linspace(2020, 2027, 100)
        scores = [MacroModel.get_cycle_status(y)[0] for y in years]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=scores, mode='lines', name='行业周期曲线', line=dict(color='#00E5FF', width=3)))
        
        # 标记当前时间点
        current_year = datetime.now().year + datetime.now().month / 12.0
        current_score, current_status = MacroModel.get_cycle_status(current_year)
        
        fig.add_trace(go.Scatter(x=[current_year], y=[current_score], mode='markers', name='当前位置', 
                                marker=dict(size=12, color='#FF3D00', symbol='diamond')))
        
        fig.update_layout(
            title="光伏行业宏观周期模型 (理论值)",
            template="plotly_dark",
            xaxis_title="年份",
            yaxis_title="景气度指数",
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Microsoft YaHei") # 尝试适配中文
        )
        return fig, current_status

# ==========================================
# 2. 情景管理器 (这里就是你的“剧本”)
# ==========================================
class ScenarioManager:
    # 这里定义了四种不同的未来剧本
    SCENARIOS = {
        "基准情形 (Base Case)": {
            "margin_shock_bps": 0, "tariff_shock_pct": 0.0, "market_demand_adj": 1.0, 
            "desc": "当前市场维持现状，无重大突发利空。"
        },
        "2025 贸易战 (严峻模式)": {
            "margin_shock_bps": 300, "tariff_shock_pct": 0.35, "market_demand_adj": 0.8, 
            "desc": "关税大幅提升至 35%，且出口受阻，毛利承压。"
        },
        "国内价格战 (内卷模式)": {
            "margin_shock_bps": 800, "tariff_shock_pct": 0.0, "market_demand_adj": 1.2, 
            "desc": "为清库存爆发惨烈价格战，全行业毛利暴跌 8%。"
        },
        "技术路线迭代 (P型淘汰)": {
            "margin_shock_bps": 200, "tariff_shock_pct": 0.05, "market_demand_adj": 0.9, 
            "desc": "旧产能被淘汰，相关资产减值风险增加。"
        }
    }

# ==========================================
# 3. 核心算法 (把财务变成概率的“榨汁机”)
# ==========================================
class CreditEnginePro:
    @staticmethod
    def sigmoid(z):
        # S型函数：把任意分数压缩到 0-1 之间
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def calculate_pd_score(row, scenario_params, macro_status):
        """
        计算逻辑：
        1. 拿原始数据
        2. 根据选定的剧本（贸易战/价格战）扣减利润
        3. 用 Logit 公式算出总分
        4. 用 Sigmoid 算出违约率 (PD)
        """
        # --- 第一步：压力测试 ---
        # 1. 毛利率冲击：比如价格战，毛利直接减去 8%
        base_gm = row['Gross Margin']
        stressed_gm = base_gm - (scenario_params['margin_shock_bps'] / 100.0)
        
        # 2. 关税冲击：只有海外收入部分会被扣税
        overseas_exposure = row['Overseas Ratio'] / 100.0
        # 关税伤害 = 海外占比 * 关税税率
        tariff_hit = overseas_exposure * scenario_params['tariff_shock_pct'] * 100
        
        # 最终的“压力后毛利率”
        final_gm = stressed_gm - tariff_hit
        
        # --- 第二步：提取其他因子 ---
        inv_days = row['Inventory Days'] # 库存天数
        cf_flag = 1 if row['Cash Flow'] > 0 else 0 # 现金流是不是正的
        
        # --- 第三步：宏观环境校准 ---
        # 如果是大环境不好（衰退/萧条），给所有人的分再扣一点
        macro_adj = 0
        if "衰退" in macro_status or "萧条" in macro_status:
            macro_adj = -0.5 
        
        # --- 第四步：Logit 评分公式 (核心) ---
        # Z = 基础分 + (权重1 * 毛利) + (权重2 * 库存) ...
        # 注意：库存越高越不好，所以系数要是正的（因为我们在算违约的概率）
        
        intercept = -1.0 
        coef_gm = -0.15      # 毛利高，违约概率低 (负号)
        coef_inv = 0.02      # 库存高，违约概率高 (正号)
        coef_cf = -1.5       # 现金流为正，违约概率大幅降低 (我是导师建议你改的权重)
        
        logit_z = intercept + (coef_gm * final_gm) + (coef_inv * inv_days) + (coef_cf * cf_flag) + macro_adj
        
        # --- 第五步：算出违约率 (PD) ---
        pd_value = CreditEnginePro.sigmoid(logit_z)
        
        # --- 第六步：换算成 0-100 的信用分 ---
        score = 100 * (1 - pd_value)
        
        # 评级映射
        if score >= 85: rating = "AAA (极好)"
        elif score >= 70: rating = "AA (优良)"
        elif score >= 55: rating = "BBB (投资级)"
        elif score >= 40: rating = "BB (投机级)"
        elif score >= 25: rating = "B (高风险)"
        else: rating = "CCC (垃圾级)"
        
        return pd.Series({
            'Stressed_GM': final_gm,
            'PD_Prob': pd_value,
            'V18_Score': score,
            'Rating': rating,
            'Logit_Z': logit_z
        })

# ==========================================
# 4. 界面渲染 (全中文)
# ==========================================
st.set_page_config(page_title="全球信贷透视系统 V18 (CN)", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    .stApp { background-color: #000000 !important; color: #E0E0E0; font-family: 'Microsoft YaHei', sans-serif; }
    [data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #333; }
    h1, h2, h3 { color: #00E5FF !important; font-weight: 800 !important; }
    .stMetric { background-color: #111; border: 1px solid #333; border-left: 4px solid #0056D2; padding: 15px; }
    </style>
""", unsafe_allow_html=True)

def main():
    # --- 侧边栏 ---
    st.sidebar.title("⚙️ 情景实验室")
    
    # 1. 宏观
    st.sidebar.markdown("### 1. 宏观周期位置")
    macro_fig, macro_status = MacroModel.plot_cycle_curve()
    st.sidebar.plotly_chart(macro_fig, use_container_width=True)
    st.sidebar.info(f"当前阶段: **{macro_status}**")
    
    # 2. 剧本选择
    st.sidebar.markdown("### 2. 压力测试剧本")
    selected_scenario_name = st.sidebar.selectbox("选择市场剧本", list(ScenarioManager.SCENARIOS.keys()))
    scenario_params = ScenarioManager.SCENARIOS[selected_scenario_name]
    
    with st.sidebar.expander("查看剧本参数详情", expanded=True):
        st.write(f"📉 毛利冲击: **{scenario_params['margin_shock_bps']} 基点**")
        st.write(f"🚢 关税冲击: **{scenario_params['tariff_shock_pct']:.0%}** (针对海外收入)")
        st.write(f"🛒 市场需求: **{scenario_params['market_demand_adj']}倍**")
        st.caption(f"📝 说明: {scenario_params['desc']}")

    # --- 主界面 ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("全球信贷透视系统 | V18.0 中文版")
        st.caption(f"基于逻辑回归 (Logistic Regression) 的动态风控模型 | 当前模式: {selected_scenario_name}")
    with c2:
        st.metric("核心算法引擎", "Logit 回归", "Sigmoid 激活")

    # --- 模拟数据 (这里你可以改成真实的) ---
    data = [
        {'Ticker': '600438.SH', 'Company Name': '通威股份 (Tongwei)', 'Gross Margin': 28.5, 'Overseas Ratio': 25.0, 'Inventory Days': 85, 'Cash Flow': 1},
        {'Ticker': '300750.SZ', 'Company Name': '宁德时代 (CATL)', 'Gross Margin': 22.0, 'Overseas Ratio': 35.0, 'Inventory Days': 70, 'Cash Flow': 1},
        {'Ticker': '688599.SH', 'Company Name': '天合光能 (Trina)', 'Gross Margin': 15.5, 'Overseas Ratio': 60.0, 'Inventory Days': 110, 'Cash Flow': 0},
        {'Ticker': '002459.SZ', 'Company Name': '晶科能源 (Jinko)', 'Gross Margin': 14.0, 'Overseas Ratio': 72.0, 'Inventory Days': 140, 'Cash Flow': 1},
        {'Ticker': '601012.SH', 'Company Name': '隆基绿能 (Longi)', 'Gross Margin': 18.0, 'Overseas Ratio': 45.0, 'Inventory Days': 95, 'Cash Flow': 1}
    ]
    df = pd.DataFrame(data)

    # --- 运行计算 ---
    res = df.apply(lambda r: CreditEnginePro.calculate_pd_score(r, scenario_params, macro_status), axis=1)
    df_final = pd.concat([df, res], axis=1)

    # --- 结果展示 ---
    
    # 核心指标卡
    k1, k2, k3, k4 = st.columns(4)
    avg_pd = df_final['PD_Prob'].mean()
    high_risk_num = len(df_final[df_final['V18_Score'] < 40])
    
    k1.metric("组合平均违约率 (PD)", f"{avg_pd:.2%}", delta="越低越好", delta_color="inverse")
    k2.metric("平均信用分", f"{df_final['V18_Score'].mean():.1f}", delta="满分100")
    k3.metric("高风险主体数", str(high_risk_num), delta="需重点关注", delta_color="inverse")
    k4.metric("宏观校准", macro_status, "已应用周期因子")

    # 图表区
    st.markdown("### 📊 风险全景图")
    t1, t2 = st.columns([2, 1])
    
    with t1:
        # S型曲线图
        x_range = np.linspace(-6, 6, 100)
        y_range = 1 / (1 + np.exp(-x_range))
        
        fig_logit = go.Figure()
        fig_logit.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='Sigmoid 曲线', line=dict(color='#444', dash='dash')))
        
        fig_logit.add_trace(go.Scatter(
            x=df_final['Logit_Z'], 
            y=df_final['PD_Prob'], 
            mode='markers+text',
            text=df_final['Company Name'],
            textposition='top center',
            marker=dict(size=12, color=df_final['V18_Score'], colorscale='RdYlGn', showscale=True),
            name='公司分布'
        ))
        
        fig_logit.update_layout(
            title="Logit 映射图 (横轴=综合得分Z, 纵轴=违约概率PD)",
            xaxis_title="Logit Z-Score (越右风险越高)",
            yaxis_title="违约概率 (PD)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig_logit, use_container_width=True)

    with t2:
        st.markdown("#### 情景冲击详情")
        # 格式化表格显示
        st.dataframe(
            df_final[['Company Name', 'Rating', 'PD_Prob', 'Stressed_GM']]
            .rename(columns={'Company Name':'公司', 'Rating':'评级', 'PD_Prob':'违约率', 'Stressed_GM':'折后毛利'})
            .style.format({'违约率': "{:.2%}", '折后毛利': "{:.1f}%"})
            .background_gradient(subset=['违约率'], cmap='Reds'),
            use_container_width=True,
            height=400
        )

    # --- PDF 导出 (保留英文，防止字体报错) ---
    st.markdown("### 📑 导出审计报告")
    st.info("注：由于PDF引擎字体限制，导出报告暂时保持英文格式。")
    
    col_pdf_sel, col_pdf_btn = st.columns([3, 1])
    target_comp = col_pdf_sel.selectbox("选择要生成报告的公司", df_final['Company Name'])
    
    if col_pdf_btn.button("生成 PDF 报告"):
        row = df_final[df_final['Company Name'] == target_comp].iloc[0]
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"CREDIT MEMO: {target_comp}", 0, 1)
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 10, f"Generated by V18.0 System | {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
        pdf.line(10, 30, 200, 30)
        pdf.ln(10)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "1. STRESS SCENARIO", 0, 1)
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 8, f"Scenario: {selected_scenario_name}", 0, 1)
        
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "2. FINANCIAL IMPACT", 0, 1)
        pdf.cell(0, 8, f"Final Score: {row['V18_Score']:.1f}", 0, 1)
        pdf.cell(0, 8, f"Implied Rating: {row['Rating']}", 0, 1)
        pdf.cell(0, 8, f"Prob. Default (PD): {row['PD_Prob']:.2%}", 0, 1)
        
        # 强制转换为 bytes，修复下载报错
        pdf_bytes = bytes(pdf.output())
        
        st.download_button(
            "📥 下载报告 (PDF)", 
            pdf_bytes, 
            f"Report_{datetime.now().strftime('%Y%m%d')}.pdf", 
            "application/pdf"
        )

if __name__ == "__main__":
    main()
