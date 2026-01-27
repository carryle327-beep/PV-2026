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
# 0. 系统配置
# ==========================================
st.set_page_config(page_title="全球信贷透视系统 V20 (数据版)", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    .stApp { background-color: #000000 !important; color: #E0E0E0; font-family: 'Microsoft YaHei', sans-serif; }
    [data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #333; }
    h1, h2, h3 { color: #00E5FF !important; font-weight: 800 !important; }
    .stMetric { background-color: #111; border: 1px solid #333; border-left: 4px solid #0056D2; padding: 15px; }
    div[data-testid="stFileUploader"] {
        border: 1px dashed #0056D2;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 业务逻辑 (保持不变，这是引擎)
# ==========================================
class CreditEngine:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def calculate(row, params, macro_status):
        # 鲁棒性处理：防止空值报错
        try:
            base_gm = float(row.get('Gross Margin', 0))
            debt_ratio = float(row.get('Debt Ratio', 50))
            overseas = float(row.get('Overseas Ratio', 0))
            inv = float(row.get('Inventory Days', 90))
            cf = float(row.get('Cash Flow', 0))
            # 现金流归一化：如果是金额，大于0记为1，否则0
            cf_flag = 1 if cf > 0 else 0
        except:
            return pd.Series({'Score': 0, 'Rating': 'Error', 'PD_Prob': 1.0})

        # 压力测试
        stressed_gm = base_gm - (params['margin_shock'] / 100.0)
        tariff_hit = (overseas / 100.0) * params['tariff_shock'] * 100
        final_gm = stressed_gm - tariff_hit
        
        # 宏观调整
        macro_adj = -0.5 if "衰退" in macro_status or "萧条" in macro_status else 0
        
        # Logit 公式
        intercept = -0.5
        logit_z = intercept + (-0.15 * final_gm) + (0.02 * inv) + (0.05 * debt_ratio) + (-1.2 * cf_flag) + macro_adj
                  
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
# 2. 数据处理工具 (Data Utils)
# ==========================================
def get_template_df():
    """生成一个标准的 Excel 模板供用户下载"""
    df = pd.DataFrame({
        'Company': ['示例公司A', '示例公司B'],
        'Gross Margin': [25.5, 15.0],
        'Overseas Ratio': [40.0, 60.0],
        'Inventory Days': [80, 120],
        'Debt Ratio': [45.0, 70.0],
        'Cash Flow': [100000, -50000]
    })
    return df

def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# ==========================================
# 3. 主程序
# ==========================================
def main():
    st.sidebar.title("🗄️ 数据控制中心")
    
    # --- 模块 A: 数据导入 (Data Ingestion) ---
    st.sidebar.subheader("1. 导入你的 52 家公司")
    uploaded_file = st.sidebar.file_uploader("上传 Excel 文件", type=['xlsx'])
    
    # 模板下载功能
    st.sidebar.markdown("---")
    st.sidebar.caption("没有标准格式？")
    template_byte = convert_df_to_excel(get_template_df())
    st.sidebar.download_button("📥 下载标准 Excel 模板", template_byte, "Data_Template.xlsx")

    # 数据加载逻辑
    if uploaded_file is not None:
        try:
            df_raw = pd.read_excel(uploaded_file)
            st.sidebar.success(f"成功加载 {len(df_raw)} 家公司数据")
        except Exception as e:
            st.error(f"文件读取失败: {e}")
            return
    else:
        # 默认演示数据 (Fallback)
        st.sidebar.info("未上传文件，使用默认演示数据")
        df_raw = pd.DataFrame([
            {'Company': '通威股份', 'Gross Margin': 28.5, 'Overseas Ratio': 25.0, 'Inventory Days': 85, 'Debt Ratio': 55.0, 'Cash Flow': 1},
            {'Company': '宁德时代', 'Gross Margin': 22.0, 'Overseas Ratio': 35.0, 'Inventory Days': 70, 'Debt Ratio': 45.0, 'Cash Flow': 1},
            {'Company': '隆基绿能', 'Gross Margin': 18.0, 'Overseas Ratio': 45.0, 'Inventory Days': 95, 'Debt Ratio': 50.0, 'Cash Flow': 1},
            {'Company': '晶科能源', 'Gross Margin': 14.0, 'Overseas Ratio': 72.0, 'Inventory Days': 140, 'Debt Ratio': 74.0, 'Cash Flow': 1},
            {'Company': '天合光能', 'Gross Margin': 15.5, 'Overseas Ratio': 60.0, 'Inventory Days': 110, 'Debt Ratio': 68.0, 'Cash Flow': 0}
        ])

    # --- 模块 B: 参数设置 ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. 设定压力剧本")
    margin_shock = st.sidebar.slider("全行业毛利冲击 (bps)", 0, 1000, 300)
    tariff_shock = st.sidebar.slider("关税冲击系数 (%)", 0.0, 1.0, 0.25)
    params = {'margin_shock': margin_shock, 'tariff_shock': tariff_shock}

    # --- 模块 C: 批量计算引擎 ---
    # 这里会对 52 家公司进行全量计算
    macro_status = "衰退期 (Down)" # 默认锁死，简化逻辑
    res = df_raw.apply(lambda r: CreditEngine.calculate(r, params, macro_status), axis=1)
    df_final = pd.concat([df_raw, res], axis=1)

    # --- 主界面 ---
    st.title("全球信贷透视系统 | V20 企业版")
    st.caption(f"当前分析主体数: {len(df_final)} 家 | 外部数据源模式")

    # --- 模块 D: 单主体搜索与报告 (Single Entity Drill-down) ---
    st.markdown("### 🔍 单主体深度穿透")
    
    # 搜索框：选择任意一家公司
    company_list = df_final['Company'].unique().tolist()
    selected_company = st.selectbox("输入或选择公司名称 (查看专属报告)", company_list)
    
    # 提取这家公司的数据
    row = df_final[df_final['Company'] == selected_company].iloc[0]

    # 展示这家公司的专属面板
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # 左侧：大大的评级卡片
        rating_color = "#28A745" if row['Score'] >= 70 else "#DC3545"
        st.markdown(f"""
            <div style="background-color:#111; padding:20px; border-left: 5px solid {rating_color};">
                <h2 style="color:white; margin:0;">{row['Company']}</h2>
                <h1 style="color:{rating_color}; font-size: 60px; margin:0;">{row['Rating']}</h1>
                <p style="color:#888;">综合得分: {row['Score']:.1f} / 100</p>
                <hr style="border-color:#333;">
                <p style="color:#CCC;">违约概率 (PD): <b>{row['PD_Prob']:.2%}</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        # 导出它的专属报告
        st.write("")
        if st.button(f"📄 导出 {selected_company} 的审计报告"):
            # PDF 生成逻辑 (简易版)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, f"CREDIT REPORT: {selected_company}", 0, 1) # 英文防止乱码
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Rating: {row['Rating']} | Score: {row['Score']:.1f}", 0, 1)
            pdf.cell(0, 10, f"Scenario Shock: -{margin_shock}bps Margin", 0, 1)
            pdf.line(10, 40, 200, 40)
            
            pdf_bytes = bytes(pdf.output())
            st.download_button("📥 点击下载 PDF", pdf_bytes, f"Report_{selected_company}.pdf", "application/pdf")

    with col2:
        # 右侧：它的雷达图或对比图
        # 1. 把它和全行业平均值对比
        avg_score = df_final['Score'].mean()
        
        fig = go.Figure()
        # 行业平均线
        fig.add_trace(go.Bar(
            y=['综合评分', '折后毛利', '负债健康度'],
            x=[avg_score, df_final['Stressed_GM'].mean(), 100-df_final['Debt Ratio'].mean()],
            name='行业平均', orientation='h', marker_color='#333'
        ))
        # 这家公司的数据
        fig.add_trace(go.Bar(
            y=['综合评分', '折后毛利', '负债健康度'],
            x=[row['Score'], row['Stressed_GM'], 100-row['Debt Ratio']],
            name=selected_company, orientation='h', marker_color='#00E5FF'
        ))
        
        fig.update_layout(title=f"{selected_company} vs 行业基准", template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # --- 模块 E: 全行业概览 (给老板看的) ---
    st.markdown("### 🌍 全行业概览 (Portfolio View)")
    t1, t2 = st.columns([2, 1])
    with t1:
        fig_map = px.treemap(df_final, path=[px.Constant("全部公司"), 'Rating', 'Company'], values='Score',
                             color='Score', color_continuous_scale='RdYlGn', title="52家公司信用热力图")
        fig_map.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_map, use_container_width=True)
    
    with t2:
        st.dataframe(df_final[['Company', 'Rating', 'Score', 'PD_Prob']].sort_values('Score'), height=400, use_container_width=True)

if __name__ == "__main__":
    main()
