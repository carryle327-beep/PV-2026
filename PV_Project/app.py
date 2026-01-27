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
# V21 系统配置 (支持代码搜索)
# ==========================================
st.set_page_config(page_title="全球信贷透视系统 V21 (代码搜索版)", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    .stApp { background-color: #000000 !important; color: #E0E0E0; font-family: 'Microsoft YaHei', sans-serif; }
    [data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #333; }
    h1, h2, h3 { color: #00E5FF !important; font-weight: 800 !important; }
    .stMetric { background-color: #111; border: 1px solid #333; border-left: 4px solid #0056D2; padding: 15px; }
    div[data-testid="stFileUploader"] { border: 1px dashed #0056D2; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 核心计算引擎 (Engine)
# ==========================================
class CreditEngine:
    @staticmethod
    def sigmoid(z):
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
            return pd.Series({'Score': 0, 'Rating': 'Error', 'PD_Prob': 1.0})

        stressed_gm = base_gm - (params['margin_shock'] / 100.0)
        tariff_hit = (overseas / 100.0) * params['tariff_shock'] * 100
        final_gm = stressed_gm - tariff_hit
        
        macro_adj = -0.5 if "衰退" in macro_status or "萧条" in macro_status else 0
        
        # Logit
        intercept = -0.5
        logit_z = intercept + (-0.15 * final_gm) + (0.02 * inv) + (0.05 * debt_ratio) + (-1.2 * cf_flag) + macro_adj
                  
        pd_val = CreditEngine.sigmoid(logit_z)
        score = 100 * (1 - pd_val)
        
        if score >= 85: rating = "AAA"
        elif score >= 70: rating = "AA"
        elif score >= 55: rating = "BBB"
        elif score >= 40: rating = "BB"
        else: rating = "CCC"
        
        return pd.Series({
            'Stressed_GM': final_gm, 'PD_Prob': pd_val, 'Score': score, 'Rating': rating
        })

# ==========================================
# 主程序 (Main)
# ==========================================
def main():
    st.sidebar.title("🗄️ 数据控制中心")
    
    # 1. 导入数据
    st.sidebar.subheader("1. 导入数据源")
    uploaded_file = st.sidebar.file_uploader("上传 Excel (需包含 Ticker 列)", type=['xlsx'])
    
    # 默认回退数据 (包含代码)
    if uploaded_file is not None:
        try:
            df_raw = pd.read_excel(uploaded_file)
            # 兼容性处理：如果用户上传了旧文件没有Ticker，给个默认值
            if 'Ticker' not in df_raw.columns:
                df_raw['Ticker'] = "N/A"
            # 确保 Ticker 是字符串
            df_raw['Ticker'] = df_raw['Ticker'].astype(str).str.replace('.0', '', regex=False)
            st.sidebar.success(f"成功加载 {len(df_raw)} 家数据")
        except Exception as e:
            st.error(f"Error: {e}")
            return
    else:
        st.sidebar.info("等待上传... (使用默认演示数据)")
        # 默认只展示几条，提醒用户上传
        df_raw = pd.DataFrame([
            {'Ticker': '600438', 'Company': '通威股份', 'Gross Margin': 28.5, 'Overseas Ratio': 25.0, 'Inventory Days': 85, 'Debt Ratio': 55.0, 'Cash Flow': 1},
            {'Ticker': '300750', 'Company': '宁德时代', 'Gross Margin': 22.0, 'Overseas Ratio': 35.0, 'Inventory Days': 70, 'Debt Ratio': 45.0, 'Cash Flow': 1}
        ])

    # 2. 参数设置
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. 压力参数")
    margin_shock = st.sidebar.slider("毛利冲击 (bps)", 0, 1000, 300)
    tariff_shock = st.sidebar.slider("关税冲击 (%)", 0.0, 1.0, 0.25)
    params = {'margin_shock': margin_shock, 'tariff_shock': tariff_shock}

    # 3. 计算
    res = df_raw.apply(lambda r: CreditEngine.calculate(r, params, "衰退期"), axis=1)
    df_final = pd.concat([df_raw, res], axis=1)
    
    # --- 构造搜索列 (Search Column) ---
    # 把代码和名称拼起来，比如 "600438 | 通威股份"
    df_final['Search_Label'] = df_final['Ticker'] + " | " + df_final['Company']

    # --- 主界面 ---
    st.title("全球信贷透视系统 | V21 代码搜索版")
    st.caption(f"已载入 {len(df_final)} 家公司 | 支持 Ticker 索引")

    # --- 搜索模块 ---
    st.markdown("### 🔍 投资标的检索")
    
    # 这里的 selectbox 现在显示的是 "代码 | 名称"
    # 用户输入 600，会自动匹配
    search_list = df_final['Search_Label'].tolist()
    selected_label = st.selectbox("输入股票代码或名称 (e.g. 600438)", search_list)
    
    # 反向提取选中的公司
    selected_ticker = selected_label.split(" | ")[0]
    row = df_final[df_final['Ticker'] == selected_ticker].iloc[0]

    # --- 详情展示 ---
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""
            <div style="background-color:#111; padding:20px; border-left: 5px solid #00E5FF;">
                <h3 style="color:#888; margin:0;">{row['Ticker']}</h3>
                <h1 style="color:white; margin:0;">{row['Company']}</h1>
                <h2 style="color:{'#28A745' if row['Score']>=70 else '#DC3545'}; margin:10px 0;">{row['Rating']}</h2>
                <p>Score: {row['Score']:.1f} | PD: {row['PD_Prob']:.2%}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        if st.button(f"📄 导出 {row['Company']} 报告"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, f"CREDIT REPORT: {row['Ticker']}", 0, 1) # 标题用代码
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Company: {row['Company']} (Simulated Name)", 0, 1) # 中文名在PDF可能乱码，这里做演示
            pdf.cell(0, 10, f"Rating: {row['Rating']} | Score: {row['Score']:.1f}", 0, 1)
            pdf.line(10, 40, 200, 40)
            pdf_bytes = bytes(pdf.output())
            st.download_button("📥 下载 PDF", pdf_bytes, f"Report_{row['Ticker']}.pdf", "application/pdf")

    with c2:
        # 雷达对比
        avg_score = df_final['Score'].mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(y=['评分', '折后毛利'], x=[avg_score, df_final['Stressed_GM'].mean()], name='行业平均', orientation='h', marker_color='#333'))
        fig.add_trace(go.Bar(y=['评分', '折后毛利'], x=[row['Score'], row['Stressed_GM']], name=row['Company'], orientation='h', marker_color='#00E5FF'))
        fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # 全局热力图
    st.markdown("---")
    st.subheader("🌍 全市场概览")
    fig_map = px.treemap(df_final, path=[px.Constant("全市场"), 'Rating', 'Search_Label'], values='Score',
                         color='Score', color_continuous_scale='RdYlGn')
    fig_map.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_map, use_container_width=True)

if __name__ == "__main__":
    main()
