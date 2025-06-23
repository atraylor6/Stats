import streamlit as st
import pandas as pd

st.title("Quick Excel Debug")

uploaded_file = st.file_uploader("Upload .xlsx", type=["xlsx"])

if uploaded_file:
    try:
        excel_file = pd.ExcelFile(uploaded_file, engine='openpyxl')
        st.write("✅ Sheets found:", excel_file.sheet_names)
    except Exception as e:
        st.error(f"❌ Excel parsing error: {e}")
