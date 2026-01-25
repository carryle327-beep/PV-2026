import streamlit as st
import pandas as pd
import os
import glob

st.set_page_config(page_title="数据侦探", layout="wide")

# 1. 自动找文件
current_folder = os.path.dirname(os.path.abspath(__file__))
xlsx_files = glob.glob(os.path.join(current_folder, "*.xlsx"))

if not xlsx_files:
    st.error("❌ 没找到Excel文件！")
    st.stop()
file_path = xlsx_files[0]

st.title("🕵️‍♂️ 数据行数大侦探")
st.write(f"正在读取文件：`{os.path.basename(file_path)}`")

# 2. 核心修复：读取 Excel 的“目录”
try:
    # 先打开 Excel "书"，看看有几章 (Sheet)
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    
    # 让用户选择读取哪一个 Sheet
    selected_sheet = st.selectbox("请选择包含完整数据的 Sheet (工作表):", sheet_names)
    
    # 读取选中的 Sheet
    # ⚠️ 注意：这里没加 cache，保证每次都读最新的
    df = pd.read_excel(file_path, sheet_name=selected_sheet)

except Exception as e:
    st.error(f"读取失败: {e}")
    st.stop()

# 3. 结果展示
real_count = len(df)
st.metric("📊 Python 实际读到的行数", f"{real_count} 行", delta=f"目标 52 行")

if real_count == 52:
    st.success("✅ 终于对上了！就是这个 Sheet！")
elif real_count == 41:
    st.warning("⚠️ 还是 41 行？请检查一下你选的 Sheet 对不对，或者 Excel 里这页是不是真的只有 41 行？")
else:
    st.info(f"读到了 {real_count} 行。")

# 4. 看看最后几行是什么 (防止最后几行被当成空值扔了)
st.write("📋 数据的最后 5 行如下 (请检查是否包含最后那几家公司):")
st.dataframe(df.tail(5))

# 5. 简单展示全部数据
st.write("📋 全部数据:")
st.dataframe(df)
