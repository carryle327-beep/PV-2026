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
# 0. 系统配置 (V25.0 终极版)
# ==========================================
st.set_page_config(page_title="Global Credit Lens V25.0", layout="wide", page_icon="🏦")

# CSS 样式: 黑金/投行风
st.markdown("""
    <style>
    /* 全局深色背景 */
    .stApp { background-color: #000000 !important; color: #E0E0E0; font-family: 'Microsoft YaHei', sans-serif; }
    
    /* 侧边栏 */
    [data-testid="stSidebar"] { background-color: #121212 !important; border-right: 1px solid #333; }
    
    /* 标题样式 */
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 700 !important; letter-spacing: 1px; }
    
    /* 指标卡片 */
    .stMetric { background-color: #1A1A1A; border: 1px solid #333; border-left: 4px solid #0056D2; padding: 15px; border-radius: 5px; }
    
    /* Tab 页签 */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1A1A1A; border-radius: 4px 4px 0 0; color: #888; }
    .stTabs [aria-selected="true"] { background-color: #0056D2 !important; color: white !important; }
    
    /* 按钮 */
    .stButton>button { background-color: #222; color: white; border: 1px solid #444; border-radius: 4px; }
    .stButton>button:hover { border-color: #0056D2; color: #0056D2; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 安全鉴权 (Authentication)
# ==========================================
def check_password():
    CORRECT_PASSWORD = "HR2026"
    def password_entered():
        if st.session_state["password"] == CORRECT_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔒 访问密钥 (Access Key)", type="password", on_change=password_entered, key="password")
        st.caption("提示: HR2026")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔒 访问密钥 (Access Key)", type="password", on_change=password_entered, key="password")
        st.error("⛔ 密钥错误")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ==========================================
# 2. 缓存加速 (Caching)
# ==========================================
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    if 'Ticker' not in df.columns: df['Ticker'] = "N/A"
    df['Ticker'] = df['Ticker'].astype(str).str.replace('.0', '', regex=False)
    return df

# ==========================================
# 3. 核心计算引擎 (Logit Engine + Stress Test)
# ==========================================
class CreditEngine:
    @staticmethod
    def sigmoid(z):
        # [算法2] Sigmoid 激活函数：将任意 Z 值压缩到 0-1 之间作为概率
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def calculate(row, params, macro_status):
        try:
            # 1. 提取基础财务指标 (Features)
            base_gm = float(row.get('Gross Margin', 0))       
            debt_ratio = float(row.get('Debt Ratio', 50))     
            overseas = float(row.get('Overseas Ratio', 0))    
            inv = float(row.get('Inventory Days', 90))        # 核心因子：库存
            cf = float(row.get('Cash Flow', 0))               # 核心因子：现金流
            cf_flag = 1 if cf > 0 else 0
        except:
            return pd.Series({'Score': 0, 'Rating': 'Error', 'PD_Prob': 1.0, 'Stressed_GM': 0})

        # 2. [算法3] 五维压力测试 (Deterministic Simulation)
        
        # A. 市场内卷
        market_hit = params['margin_shock'] / 100.0
        # B. 关税壁垒
        tariff_hit = (overseas / 100.0) * params['tariff_shock'] * 100
        # C. 原材料通胀
        input_cost_hit = params['raw_material_shock'] * 0.2
        # D. 汇率波动
        fx_hit = (overseas / 100.0) * params['fx_shock'] 

        # 中间变量：折后毛利
        final_gm = base_gm - market_hit - tariff_hit - input_cost_hit - fx_hit
        final_gm = max(final_gm, -10.0) # 兜底逻辑

        # E. 加息冲击 (针对高负债的惩罚)
        rate_hit = (debt_ratio / 100.0) * (params['rate_hike_bps'] / 100.0) * 5.0

        # 3. [算法1] Logit 评分模型 (Linear Weighting)
        intercept = -0.5
        logit_z = intercept + \
                  (-0.15 * final_gm) + \
                  (0.02 * inv) + \
                  (0.05 * debt_ratio) + \
                  (-1.2 * cf_flag) + \
                  rate_hit
                  
        pd_val = CreditEngine.sigmoid(logit_z)
        score = 100 * (1 - pd_val)
        
        # 4. 评级映射
        if score >= 85: rating = "AAA"
        elif score >= 70: rating = "AA"
        elif score >= 55: rating = "BBB"
        elif score >= 40: rating = "BB"
        else: rating = "CCC"
        
        return pd.Series({
            'Stressed_GM': final_gm, 
            'PD_Prob': pd_val, 
            'Score': score, 
            'Rating': rating
        })

# ==========================================
# 4. IV 计算引擎 (Feature Selection Engine)
# ==========================================
class IV_Engine:
    @staticmethod
    def calculate_iv(df, target_col='Is_Bad', feature_cols=[]):
        """
        自动计算指定特征的 IV 值，用于验证因子的预测力。
        算法逻辑：分箱 -> 计数 -> WOE计算 -> IV汇总
        """
        iv_list = []
        
        for col in feature_cols:
            try:
                # 数据预处理
                temp_df = df[[col, target_col]].copy()
                temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0)
                
                # 1. 自动分箱 (Binning) - 优先用 qcut (等频)，失败用 cut (等宽)
                try:
                    temp_df['bucket'] = pd.qcut(temp_df[col], q=4, duplicates='drop')
                except:
                    temp_df['bucket'] = pd.cut(temp_df[col], bins=4)
                
                # 2. 统计好坏样本 (Aggregation)
                grouped = temp_df.groupby('bucket', observed=False)[target_col].agg(['count', 'sum'])
                grouped['bad'] = grouped['sum']
                grouped['good'] = grouped['count'] - grouped['sum']
                
                # 3. 平滑处理 (Smoothing) 防止除以0
                total_bad = grouped['bad'].sum() + 1e-5
                total_good = grouped['good'].sum() + 1e-5
                
                # 4. 计算 WOE 和 IV
                grouped['dist_bad'] = (grouped['bad'] + 1e-5) / total_bad
                grouped['dist_good'] = (grouped['good'] + 1e-5) / total_good
                grouped['woe'] = np.log(grouped['dist_good'] / grouped['dist_bad'])
                grouped['iv'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['woe']
                
                total_iv = grouped['iv'].sum()
                
                iv_list.append({'Feature': col, 'IV': total_iv})
                
            except Exception as e:
                continue # 如果某列计算失败，跳过
                
        # 返回按 IV 降序排列的结果
        return pd.DataFrame(iv_list).sort_values(by='IV', ascending=False)

# ==========================================
# 5. 主程序 (Main Application)
# ==========================================
def main():
    st.sidebar.title("🎛️ 压力测试实验室")
    
    # --- A. 数据接入 ---
    st.sidebar.subheader("1. 数据接入 (Data Feed)")
    uploaded_file = st.sidebar.file_uploader("上传 Excel", type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            df_raw = load_data(uploaded_file)
            st.sidebar.success(f"已联网: {len(df_raw)} 家主体")
        except: return
    else:
        st.sidebar.info("使用演示数据...")
        # 演示数据
        df_raw = pd.DataFrame([
            {'Ticker': '600438', 'Company': '通威股份', 'Gross Margin': 28.5, 'Overseas Ratio': 25.0, 'Inventory Days': 85, 'Debt Ratio': 55.0, 'Cash Flow': 1},
            {'Ticker': '300750', 'Company': '宁德时代', 'Gross Margin': 22.0, 'Overseas Ratio': 35.0, 'Inventory Days': 70, 'Debt Ratio': 45.0, 'Cash Flow': 1},
            {'Ticker': '601012', 'Company': '隆基绿能', 'Gross Margin': 18.0, 'Overseas Ratio': 45.0, 'Inventory Days': 95, 'Debt Ratio': 50.0, 'Cash Flow': 1},
            {'Ticker': '688599', 'Company': '天合光能', 'Gross Margin': 16.0, 'Overseas Ratio': 60.0, 'Inventory Days': 80, 'Debt Ratio': 65.0, 'Cash Flow': 1},
            {'Ticker': '002459', 'Company': '晶澳科技', 'Gross Margin': 15.5, 'Overseas Ratio': 55.0, 'Inventory Days': 88, 'Debt Ratio': 60.0, 'Cash Flow': 0}
        ])

    # --- B. 五维压力参数 ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. 宏观压力参数 (5 Factors)")
    
    st.sidebar.caption("📉 市场环境")
    margin_shock = st.sidebar.slider("1. 行业内卷 (bps)", 0, 1000, 300)
    
    st.sidebar.caption("🚢 地缘政治")
    tariff_shock = st.sidebar.slider("2. 关税壁垒 (%)", 0.0, 1.0, 0.25)
    
    st.sidebar.caption("💰 资金成本")
    rate_hike = st.sidebar.slider("3. 美联储加息 (bps)", 0, 500, 100)
    
    st.sidebar.caption("🧱 供应链")
    raw_mat_shock = st.sidebar.slider("4. 原材料通胀 (%)", 0, 50, 10)
    
    st.sidebar.caption("💱 汇率风险")
    fx_shock = st.sidebar.slider("5. 汇率波动损失 (%)", 0, 20, 5)
    
    params = {
        'margin_shock': margin_shock, 
        'tariff_shock': tariff_shock,
        'rate_hike_bps': rate_hike,
        'raw_material_shock': raw_mat_shock,
        'fx_shock': fx_shock
    }

    # --- C. 批量计算 ---
    try:
        res = df_raw.apply(lambda r: CreditEngine.calculate(r, params, "衰退期"), axis=1)
        df_final = pd.concat([df_raw, res], axis=1)
        df_final['Search_Label'] = df_final['Ticker'] + " | " + df_final['Company']
    except: return

    # ==========================================
    # 前端展示层 (Visualization Layer)
    # ==========================================
    st.title("GLOBAL CREDIT LENS | V25.0")
    st.caption(f"架构: Logit + 5-Factor Stress + IV Analysis | 样本: {len(df_final)}")
    
    # 1. 穿透式检索
    search_list = df_final['Search_Label'].tolist()
    c_search, c_blank = st.columns([1, 2])
    with c_search:
        selected_label = st.selectbox("🔍 穿透式检索 (Ticker/Name)", search_list)
    
    selected_ticker = selected_label.split(" | ")[0]
    row = df_final[df_final['Ticker'] == selected_ticker].iloc[0]

    # 2. 单体画像卡片
    col1, col2 = st.columns([1, 2])
    with col1:
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
        
        # PDF 报告生成
        if st.button(f"📄 导出 {row['Ticker']} 研报"):
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 24)
                pdf.cell(0, 20, f"CREDIT MEMO: {row['Ticker']}", 0, 1, 'C')
                pdf.line(10, 30, 200, 30)
                pdf.ln(10)
                
                pdf.set_font("Arial", "", 12)
                pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
                pdf.cell(0, 10, f"Rating: {str(row['Rating']).split(' ')[0]}", 0, 1)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"PD: {row['PD_Prob']:.2%}", 0, 1)
                
                pdf.ln(10)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "5-FACTOR STRESS TEST:", 0, 1)
                pdf.set_font("Arial", "", 11)
                pdf.cell(0, 8, f"1. Margin Shock: -{params['margin_shock']} bps", 0, 1)
                pdf.cell(0, 8, f"2. Tariff Shock: -{params['tariff_shock']*100:.0f}%", 0, 1)
                pdf.cell(0, 8, f"3. Rate Hike: +{params['rate_hike_bps']} bps", 0, 1)
                pdf.cell(0, 8, f"4. Input Cost: +{params['raw_material_shock']}%", 0, 1)
                pdf.cell(0, 8, f"5. FX Shock: -{params['fx_shock']}%", 0, 1)
                
                pdf.ln(10)
                pdf.set_font("Arial", "I", 10)
                pdf.cell(0, 10, "Note: Generated by Global Credit Lens V25.0", 0, 1)
                
                pdf_bytes = bytes(pdf.output())
                st.download_button("📥 下载 PDF", pdf_bytes, f"Report_{row['Ticker']}.pdf", "application/pdf")
            except Exception as e:
                st.error(f"导出失败: {e}")

    with col2:
        # 五维雷达图
        categories = ['综合评分', '毛利抗压', '负债健康', '现金流', '库存周转']
        def normalize(val, max_val): return min(max(val, 0), max_val) / max_val * 100
        
        row_vals = [
            row['Score'], 
            normalize(row['Stressed_GM'] + 10, 50), 
            normalize(100 - row['Debt Ratio'], 100),
            100 if row['Cash Flow'] > 0 else 20,
            normalize(365 - row['Inventory Days'], 365)
        ]
        
        avg_vals = [
            df_final['Score'].mean(),
            normalize(df_final['Stressed_GM'].mean() + 10, 50),
            normalize(100 - df_final['Debt Ratio'].mean(), 100),
            60,
            normalize(365 - df_final['Inventory Days'].mean(), 365)
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=avg_vals, theta=categories, fill='toself', name='行业平均', line_color='#444'))
        fig.add_trace(go.Scatterpolar(r=row_vals, theta=categories, fill='toself', name=row['Company'], line_color='#00E5FF'))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            template="plotly_dark", height=320, 
            title=f"{row['Company']} 五维健康度雷达",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # 宏观看板：全市场 + 因子分析
    # ==========================================
    st.subheader("📊 深度量化看板 (Portfolio Analytics)")
    
    # 包含 5 个 Tab：热力图、气泡图、分布、相关性、IV筛选
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ 全景热力图", "🛁 竞争格局", "🎻 评级分布", "🔗 归因分析", "🧠 因子筛选(IV)"])

    # 1. 热力图
    with tab1:
        if not df_final.empty:
            fig_map = px.treemap(df_final, path=[px.Constant("全市场"), 'Rating', 'Search_Label'], values='Score',
                                 color='Score', color_continuous_scale='RdYlGn', title="信用风险分布热力图")
            fig_map.update_layout(template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_map, use_container_width=True)

    # 2. 气泡图
    with tab2:
        if not df_final.empty:
            fig_bub = px.scatter(df_final, x="Stressed_GM", y="Score", size="Debt Ratio", color="Rating",
                                 hover_name="Company", text="Company", title="盈利能力 vs 信用评分",
                                 color_discrete_sequence=px.colors.qualitative.Bold)
            fig_bub.update_layout(template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bub, use_container_width=True)
    
    # 3. 分布图
    with tab3:
        if not df_final.empty:
            fig_vio = px.strip(df_final, x="Rating", y="Score", color="Rating", title="信用评级分布")
            fig_vio.update_layout(template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_vio, use_container_width=True)
    
    # 4. 相关性
    with tab4:
        if not df_final.empty:
            cols_to_corr = ['Score', 'Gross Margin', 'Overseas Ratio', 'Inventory Days', 'Debt Ratio']
            corr_matrix = df_final[cols_to_corr].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="风险因子相关性")
            fig_corr.update_layout(template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_corr, use_container_width=True)

    # 5. IV 因子筛选 (核心升级功能)
    with tab5:
        st.markdown("### 🧬 特征重要性分析 (Information Value)")
        st.caption("Auto-calculated using WOE/IV Engine. Identifies top predictive factors.")
        
        if not df_final.empty:
            # =================================================
            # 核心逻辑：Ground Truth (真实标签) vs Proxy (影子标签)
            # =================================================
            if 'Manual_Bad_Label' in df_final.columns:
                st.success("✅ 检测到人工标注的真实违约数据 (Ground Truth)，正在计算真实 IV...")
                target_col = 'Manual_Bad_Label'
            else:
                st.warning("⚠️ 未检测到真实违约标签，正在使用模型预测值 (Proxy Label) 进行逻辑自洽性验证...")
                # 影子变量逻辑：假设 PD > 30% 为高风险
                df_final['Is_Bad'] = df_final['PD_Prob'].apply(lambda x: 1 if x > 0.30 else 0)
                target_col = 'Is_Bad'
            
            # 定义需要分析的因子
            feature_cols = ['Gross Margin', 'Debt Ratio', 'Overseas Ratio', 'Inventory Days', 'Cash Flow']
            
            # 调用引擎
            iv_result = IV_Engine.calculate_iv(df_final, target_col=target_col, feature_cols=feature_cols)
            
            c_iv1, c_iv2 = st.columns([2, 1])
            with c_iv1:
                # 动态着色：强因子(>0.3)显示金色，中等显示蓝色
                iv_result['Color'] = iv_result['IV'].apply(lambda x: '#FFD700' if x > 0.3 else ('#00E5FF' if x > 0.1 else '#555555'))
                
                fig_iv = px.bar(iv_result, x='IV', y='Feature', orientation='h', 
                                title="关键风险因子预测力排行 (IV Value)",
                                text_auto='.3f',
                                color='Feature', 
                                color_discrete_map={row['Feature']: row['Color'] for _, row in iv_result.iterrows()})
                
                fig_iv.update_layout(template="plotly_dark", height=400, showlegend=False,
                                     xaxis_title="Information Value (IV)", yaxis_title="Risk Factors")
                st.plotly_chart(fig_iv, use_container_width=True)
            
            with c_iv2:
                st.info("💡 **IV 阈值标准:**\n\n- **> 0.3 (Gold)**: Strong Predictor (核心因子)\n- **0.1 - 0.3**: Medium Predictor (有效因子)\n- **< 0.02**: Useless (噪音)")
                st.dataframe(iv_result[['Feature', 'IV']], use_container_width=True)

if __name__ == "__main__":
    main()
