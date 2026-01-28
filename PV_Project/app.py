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
# 0. 系统配置 (旗舰版黑白风 - V22 UI)
# ==========================================
st.set_page_config(page_title="全球信贷透视系统 V22.1 (旗舰版)", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    /* 全局黑底 */
    .stApp { background-color: #000000 !important; color: #E0E0E0; font-family: 'Microsoft YaHei', sans-serif; }
    
    /* 侧边栏深灰 */
    [data-testid="stSidebar"] { background-color: #121212 !important; border-right: 1px solid #333; }
    
    /* 标题改为纯白，更显高级与冷峻 */
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 700 !important; letter-spacing: 1px; }
    
    /* 指标卡样式：黑底白字，左侧保留一点点蓝作为点缀 */
    .stMetric { background-color: #1A1A1A; border: 1px solid #333; border-left: 4px solid #0056D2; padding: 15px; border-radius: 5px; }
    
    /* Tab页签样式 */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1A1A1A; border-radius: 4px 4px 0 0; color: #888; }
    .stTabs [aria-selected="true"] { background-color: #0056D2 !important; color: white !important; }
    
    /* 按钮样式 */
    .stButton>button { background-color: #222; color: white; border: 1px solid #444; border-radius: 4px; }
    .stButton>button:hover { border-color: #0056D2; color: #0056D2; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 核心计算引擎 (Engine)
# ==========================================
class CreditEngine:
    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -10, 10)
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
            return pd.Series({'Score': 0, 'Rating': 'Error', 'PD_Prob': 1.0, 'Stressed_GM': 0})

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
            'Stressed_GM': final_gm, 'PD_Prob': pd_val, 'Score': score, 'Rating': rating
        })

# ==========================================
# 2. 主程序 (Main)
# ==========================================
def main():
    st.sidebar.title("🎛️ 风控控制台")
    
    # --- A. 数据源 ---
    st.sidebar.subheader("1. 数据接入")
    uploaded_file = st.sidebar.file_uploader("上传 Excel", type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            df_raw = pd.read_excel(uploaded_file)
            if 'Ticker' not in df_raw.columns: df_raw['Ticker'] = "N/A"
            df_raw['Ticker'] = df_raw['Ticker'].astype(str).str.replace('.0', '', regex=False)
            st.sidebar.success(f"已联网: {len(df_raw)} 家主体")
        except:
            return
    else:
        # 默认数据 (演示用)
        st.sidebar.info("使用演示数据...")
        df_raw = pd.DataFrame([
            {'Ticker': '600438', 'Company': '通威股份', 'Gross Margin': 28.5, 'Overseas Ratio': 25.0, 'Inventory Days': 85, 'Debt Ratio': 55.0, 'Cash Flow': 1},
            {'Ticker': '300750', 'Company': '宁德时代', 'Gross Margin': 22.0, 'Overseas Ratio': 35.0, 'Inventory Days': 70, 'Debt Ratio': 45.0, 'Cash Flow': 1},
            {'Ticker': '601012', 'Company': '隆基绿能', 'Gross Margin': 18.0, 'Overseas Ratio': 45.0, 'Inventory Days': 95, 'Debt Ratio': 50.0, 'Cash Flow': 1}
        ])

    # --- B. 参数 ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. 压力参数")
    margin_shock = st.sidebar.slider("毛利冲击 (bps)", 0, 1000, 300)
    tariff_shock = st.sidebar.slider("关税冲击 (%)", 0.0, 1.0, 0.25)
    params = {'margin_shock': margin_shock, 'tariff_shock': tariff_shock}

    # --- 计算 ---
    try:
        res = df_raw.apply(lambda r: CreditEngine.calculate(r, params, "衰退期"), axis=1)
        df_final = pd.concat([df_raw, res], axis=1)
        df_final['Search_Label'] = df_final['Ticker'] + " | " + df_final['Company']
    except:
        return

    # ==========================================
    # 界面第一部分：单体穿透 (Micro View)
    # ==========================================
    st.title("GLOBAL CREDIT LENS | V22.1")
    st.caption(f"当前分析样本: {len(df_final)} 家 | 模式: 压力测试 (Stress Testing)")
    
    # 搜索条
    search_list = df_final['Search_Label'].tolist()
    c_search, c_blank = st.columns([1, 2])
    with c_search:
        selected_label = st.selectbox("🔍 穿透式检索 (Ticker/Name)", search_list)
    
    selected_ticker = selected_label.split(" | ")[0]
    row = df_final[df_final['Ticker'] == selected_ticker].iloc[0]

    # 单体展示区
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # 评级卡片
        rating_color = '#28A745' if row['Score'] >= 70 else '#DC3545'
        st.markdown(f"""
            <div style="background-color:#1A1A1A; padding:20px; border-radius:8px; border:1px solid #333;">
                <h4 style="color:#888; margin:0;">{row['Ticker']}</h4>
                <h2 style="color:white; margin:5px 0;">{row['Company']}</h2>
                <div style="margin-top:15px; padding:10px; background-color:{rating_color}20; border-left:4px solid {rating_color};">
                    <h1 style="color:{rating_color}; margin:0; font-size:48px;">{row['Rating']}</h1>
                </div>
                <p style="color:#AAA; margin-top:10px;">Score: <b>{row['Score']:.1f}</b> | PD: <b>{row['PD_Prob']:.2%}</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        # --- 满血复活的 PDF 导出功能 (V21 内核) ---
        if st.button(f"📄 导出 {row['Ticker']} 完整审计报告"):
            try:
                pdf = FPDF()
                pdf.add_page()
                
                # 1. 标题回归专业风 (CREDIT MEMO)
                pdf.set_font("Arial", "B", 24)
                pdf.cell(0, 20, f"CREDIT MEMO: {row['Ticker']}", 0, 1, 'C')
                pdf.line(10, 30, 200, 30)
                pdf.ln(10)
                
                # 2. 核心数据 (找回 PD)
                pdf.set_font("Arial", "", 12)
                pdf.cell(0, 10, f"Report Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
                pdf.cell(0, 10, f"Internal Rating: {str(row['Rating']).split(' ')[0]}", 0, 1)
                pdf.cell(0, 10, f"Credit Score: {row['Score']:.1f} / 100", 0, 1)
                
                # 加粗显示违约概率 (PD)
                pdf.set_font("Arial", "B", 12) 
                pdf.cell(0, 10, f"Probability of Default (PD): {row['PD_Prob']:.2%}", 0, 1) 
                
                pdf.ln(10)
                
                # 3. 压力参数详情 (找回关税 Tariff)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "STRESS TEST SCENARIO:", 0, 1)
                pdf.set_font("Arial", "", 11)
                pdf.cell(0, 8, f"- Margin Shock: -{params['margin_shock']} bps (Profit Impact)", 0, 1)
                pdf.cell(0, 8, f"- Tariff Shock: -{params['tariff_shock']*100:.0f}% (Overseas Impact)", 0, 1) 
                
                pdf.ln(10)
                pdf.set_font("Arial", "I", 10)
                pdf.cell(0, 10, "Note: Company name omitted for universal encoding compatibility.", 0, 1)
                
                # 4. 生成文件
                pdf_bytes = bytes(pdf.output())
                st.download_button("📥 下载文件", pdf_bytes, f"Credit_Memo_{row['Ticker']}.pdf", "application/pdf")
            except Exception as e:
                st.error(f"导出失败: {e}")

    with col2:
        # 对比图 (Benchmark)
        avg_score = df_final['Score'].mean()
        avg_gm = df_final['Stressed_GM'].mean()
        
        fig = go.Figure()
        # 行业线
        fig.add_trace(go.Bar(
            y=['综合评分', '压力后毛利', '负债健康度(1-Debt%)'], 
            x=[avg_score, avg_gm, 100-df_final['Debt Ratio'].mean()],
            name='行业平均', orientation='h', marker_color='#333'
        ))
        # 个股线
        fig.add_trace(go.Bar(
            y=['综合评分', '压力后毛利', '负债健康度(1-Debt%)'], 
            x=[row['Score'], row['Stressed_GM'], 100-row['Debt Ratio']],
            name=row['Company'], orientation='h', marker_color='#00E5FF' # 个股保留高亮蓝，突出显示
        ))
        fig.update_layout(
            title=f"{row['Company']} vs 行业基准", 
            template="plotly_dark", 
            height=320, 
            margin=dict(l=0,r=0,t=40,b=0),
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # 界面第二部分：深度量化看板 (Macro View)
    # ==========================================
    st.subheader("📊 深度量化看板 (Portfolio Analytics)")
    
    # 4个Tab全保留
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ 全景热力图", "🛁 竞争格局(气泡)", "🎻 评级分布", "🔗 归因分析"])

    # 1. 热力图 (Treemap)
    with tab1:
        if not df_final.empty:
            fig_map = px.treemap(df_final, path=[px.Constant("全市场"), 'Rating', 'Search_Label'], values='Score',
                                 color='Score', color_continuous_scale='RdYlGn',
                                 title="信用风险分布热力图 (面积=评分权重)")
            fig_map.update_layout(template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_map, use_container_width=True)

    # 2. 气泡图 (Bubble)
    with tab2:
        if not df_final.empty:
            # X轴=毛利, Y轴=评分, 大小=负债率
            fig_bub = px.scatter(df_final, x="Stressed_GM", y="Score", size="Debt Ratio", color="Rating",
                                 hover_name="Company", text="Company",
                                 title="盈利能力 vs 信用评分 (气泡大小=负债率)",
                                 labels={"Stressed_GM": "压力后毛利率(%)", "Score": "信用评分", "Debt Ratio": "负债率"},
                                 color_discrete_sequence=px.colors.qualitative.Bold)
            fig_bub.update_traces(textposition='top center')
            fig_bub.update_layout(template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bub, use_container_width=True)
            st.caption("💡 洞察：位于右下角的大气泡是'高风险僵尸企业'（负债高、分低），右上角是'现金牛'。")

    # 3. 分布图 (Strip/Violin)
    with tab3:
        if not df_final.empty:
            fig_vio = px.strip(df_final, x="Rating", y="Score", color="Rating", 
                               title="信用评级分布密度",
                               category_orders={"Rating": ["AAA", "AA", "BBB", "BB", "CCC"]})
            fig_vio.update_layout(template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_vio, use_container_width=True)
            st.caption("💡 洞察：观察点的密集程度。如果大量点集中在 CCC，说明行业系统性风险极高。")

    # 4. 相关性矩阵 (Correlation)
    with tab4:
        if not df_final.empty:
            # 只选取数值型列进行计算
            cols_to_corr = ['Score', 'Gross Margin', 'Overseas Ratio', 'Inventory Days', 'Debt Ratio']
            corr_matrix = df_final[cols_to_corr].corr()
            
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                 color_continuous_scale='RdBu_r', 
                                 title="风险因子相关性矩阵 (Factor Correlation)")
            fig_corr.update_layout(template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("💡 洞察：红色(1.0)代表正相关，蓝色(-1.0)代表负相关。查看哪个因子对 Score 的影响最大（颜色最深）。")

if __name__ == "__main__":
    main()
