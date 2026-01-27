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
# 0. 系统配置 (黑金样式)
# ==========================================
st.set_page_config(page_title="全球信贷透视系统 V21 (稳定版)", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    /* 全局黑底白字 */
    .stApp { background-color: #000000 !important; color: #E0E0E0; font-family: 'Microsoft YaHei', sans-serif; }
    
    /* 侧边栏深灰 */
    [data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #333; }
    
    /* 标题高亮 */
    h1, h2, h3 { color: #00E5FF !important; font-weight: 800 !important; }
    
    /* 指标卡样式 */
    .stMetric { background-color: #111; border: 1px solid #333; border-left: 4px solid #0056D2; padding: 15px; }
    
    /* 上传框样式 */
    div[data-testid="stFileUploader"] { border: 1px dashed #0056D2; padding: 10px; border-radius: 5px; }
    
    /* 按钮样式 */
    .stButton>button { background-color: #0056D2; color: white; border: none; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 核心计算引擎 (Engine)
# ==========================================
class CreditEngine:
    @staticmethod
    def sigmoid(z):
        # 限制 z 的范围，防止溢出
        z = np.clip(z, -10, 10)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def calculate(row, params, macro_status):
        # 鲁棒性处理：防止空值导致计算崩溃
        try:
            base_gm = float(row.get('Gross Margin', 0))
            debt_ratio = float(row.get('Debt Ratio', 50))
            overseas = float(row.get('Overseas Ratio', 0))
            inv = float(row.get('Inventory Days', 90))
            cf = float(row.get('Cash Flow', 0))
            cf_flag = 1 if cf > 0 else 0
        except:
            # 如果数据有问题，返回默认安全值
            return pd.Series({'Score': 0, 'Rating': 'Error', 'PD_Prob': 1.0, 'Stressed_GM': 0})

        # 压力测试计算
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
        
        # 评级映射
        if score >= 85: rating = "AAA"
        elif score >= 70: rating = "AA"
        elif score >= 55: rating = "BBB"
        elif score >= 40: rating = "BB"
        else: rating = "CCC"
        
        return pd.Series({
            'Stressed_GM': final_gm, 'PD_Prob': pd_val, 'Score': score, 'Rating': rating
        })

# ==========================================
# 2. 主程序 (Main)
# ==========================================
def main():
    st.sidebar.title("🗄️ 数据控制中心")
    
    # --- 模块 A: 数据导入 ---
    st.sidebar.subheader("1. 导入数据源")
    uploaded_file = st.sidebar.file_uploader("上传 Excel (需包含 Ticker 列)", type=['xlsx'])
    
    # 数据加载逻辑
    if uploaded_file is not None:
        try:
            df_raw = pd.read_excel(uploaded_file)
            # 兼容性处理
            if 'Ticker' not in df_raw.columns:
                df_raw['Ticker'] = "N/A"
            df_raw['Ticker'] = df_raw['Ticker'].astype(str).str.replace('.0', '', regex=False)
            st.sidebar.success(f"成功加载 {len(df_raw)} 家数据")
        except Exception as e:
            st.error(f"文件读取错误: {e}")
            return # 停止运行，防止黑屏
    else:
        st.sidebar.info("等待上传... (使用演示数据)")
        # 演示数据
        df_raw = pd.DataFrame([
            {'Ticker': '600438', 'Company': '通威股份', 'Gross Margin': 28.5, 'Overseas Ratio': 25.0, 'Inventory Days': 85, 'Debt Ratio': 55.0, 'Cash Flow': 1},
            {'Ticker': '300750', 'Company': '宁德时代', 'Gross Margin': 22.0, 'Overseas Ratio': 35.0, 'Inventory Days': 70, 'Debt Ratio': 45.0, 'Cash Flow': 1}
        ])

    # --- 模块 B: 参数设置 ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. 压力参数")
    margin_shock = st.sidebar.slider("毛利冲击 (bps)", 0, 1000, 300)
    tariff_shock = st.sidebar.slider("关税冲击 (%)", 0.0, 1.0, 0.25)
    params = {'margin_shock': margin_shock, 'tariff_shock': tariff_shock}

    # --- 模块 C: 计算 ---
    # 使用 try-except 包裹计算过程，防止黑屏
    try:
        res = df_raw.apply(lambda r: CreditEngine.calculate(r, params, "衰退期"), axis=1)
        df_final = pd.concat([df_raw, res], axis=1)
        
        # 构造搜索列
        df_final['Search_Label'] = df_final['Ticker'] + " | " + df_final['Company']
    except Exception as e:
        st.error(f"计算引擎故障: {e}")
        return

    # --- 主界面渲染 ---
    st.title("全球信贷透视系统 | V21.0")
    st.caption(f"已载入 {len(df_final)} 家公司 | 支持代码 (Ticker) 索引")

    # --- 搜索模块 ---
    st.markdown("### 🔍 投资标的检索")
    
    search_list = df_final['Search_Label'].tolist()
    if search_list:
        selected_label = st.selectbox("输入股票代码或名称 (e.g. 600)", search_list)
        selected_ticker = selected_label.split(" | ")[0]
        row = df_final[df_final['Ticker'] == selected_ticker].iloc[0]

        # --- 详情展示 ---
        c1, c2 = st.columns([1, 2])
        with c1:
            rating_color = '#28A745' if row['Score'] >= 70 else '#DC3545'
            st.markdown(f"""
                <div style="background-color:#111; padding:20px; border-left: 5px solid #00E5FF;">
                    <h3 style="color:#888; margin:0;">{row['Ticker']}</h3>
                    <h1 style="color:white; margin:0;">{row['Company']}</h1>
                    <h2 style="color:{rating_color}; margin:10px 0;">{row['Rating']}</h2>
                    <p style="color:#AAA;">Score: {row['Score']:.1f} | PD: {row['PD_Prob']:.2%}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            
            # === 这里是修复后的 PDF 导出逻辑 ===
            if st.button(f"📄 导出 {row['Ticker']} 报告"):
                try:
                    # 1. 准备安全数据 (纯英文/数字)
                    ticker_safe = str(row['Ticker']).strip()
                    score_safe = f"{row['Score']:.1f}"
                    pd_safe = f"{row['PD_Prob']:.2%}"
                    rating_safe = str(row['Rating']).split(' ')[0] # 去掉中文备注
                    
                    # 2. 生成 PDF
                    pdf = FPDF()
                    pdf.add_page()
                    
                    # 标题
                    pdf.set_font("Arial", "B", 24)
                    pdf.cell(0, 20, f"CREDIT MEMO: {ticker_safe}", 0, 1, 'C')
                    pdf.line(10, 30, 200, 30)
                    pdf.ln(10)
                    
                    # 正文
                    pdf.set_font("Arial", "", 12)
                    pdf.cell(0, 10, f"Report Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
                    pdf.cell(0, 10, f"Credit Score: {score_safe} / 100", 0, 1)
                    pdf.cell(0, 10, f"Internal Rating: {rating_safe}", 0, 1)
                    pdf.cell(0, 10, f"Probability of Default: {pd_safe}", 0, 1)
                    
                    pdf.ln(20)
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, "STRESS TEST PARAMETERS:", 0, 1)
                    pdf.set_font("Arial", "", 10)
                    pdf.cell(0, 10, f"- Margin Shock: -{params['margin_shock']} bps", 0, 1)
                    pdf.cell(0, 10, f"- Tariff Shock: -{params['tariff_shock']*100:.0f}%", 0, 1)

                    pdf.ln(10)
                    pdf.set_font("Arial", "I", 10)
                    pdf.cell(0, 10, "Note: Company name omitted for universal encoding compatibility.", 0, 1)

                    # 3. 输出二进制 (bytes)
                    pdf_bytes = bytes(pdf.output())
                    
                    # 4. 下载按钮
                    st.download_button(
                        label="📥 下载英文报告 (PDF)",
                        data=pdf_bytes,
                        file_name=f"Report_{ticker_safe}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"PDF生成失败: {e}")
            # =================================

        with c2:
            # 简单的雷达/条形图对比
            avg_score = df_final['Score'].mean()
            fig = go.Figure()
            fig.add_trace(go.Bar(y=['综合评分', '折后毛利'], x=[avg_score, df_final['Stressed_GM'].mean()], name='行业平均', orientation='h', marker_color='#333'))
            fig.add_trace(go.Bar(y=['综合评分', '折后毛利'], x=[row['Score'], row['Stressed_GM']], name=row['Company'], orientation='h', marker_color='#00E5FF'))
            fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    # 全局热力图
    st.markdown("---")
    st.subheader("🌍 全市场概览")
    if not df_final.empty:
        fig_map = px.treemap(df_final, path=[px.Constant("全市场"), 'Rating', 'Search_Label'], values='Score',
                             color='Score', color_continuous_scale='RdYlGn')
        fig_map.update_layout(template="plotly_dark", height=450, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_map, use_container_width=True)

if __name__ == "__main__":
    main()
