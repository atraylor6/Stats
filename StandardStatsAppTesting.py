import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from scipy.stats import skew

# Dummy fallback for average 10-year yield
def average10Yr_dummy():
    return 3.5  # You can later replace with FRED API

# ---- Stat functions ----

def annualizedReturn(returns):
    productReturns = (1 + returns).prod()
    totalRows = returns.shape[0]
    geometricMeanReturn = (productReturns ** (12 / totalRows)) - 1
    return geometricMeanReturn * 100

def meanExcessReturn(returns, benchmarkColumn):
    columnAnnualized = returns.apply(annualizedReturn)
    excessReturn = columnAnnualized - columnAnnualized[benchmarkColumn]
    excessReturn[benchmarkColumn] = np.nan
    return excessReturn

def annualizedStandardDeviation(returns):
    monthlyStd = returns.std()
    annualStd = (monthlyStd * (12 ** 0.5)) * 100
    return annualStd

def sharpeRatio(returns, avg10yr):
    ann_return = returns.apply(annualizedReturn)
    excess = ann_return - avg10yr
    std_dev = annualizedStandardDeviation(returns)
    return excess / std_dev

def standardStats(df, benchmarkColumn):
    returns = df.drop(columns=["signal"]) if "signal" in df.columns else df.copy()
    returns.index = pd.to_datetime(returns.index)
    avg10 = average10Yr_dummy()
    stats = pd.DataFrame({
        "Annualized Return": annualizedReturn(returns),
        "Excess Return": meanExcessReturn(returns, benchmarkColumn),
        "Standard Deviation": annualizedStandardDeviation(returns),
        "Sharpe Ratio": sharpeRatio(returns, avg10)
    })
    return stats

# ---- Excel Formatting ----

def to_excel_formatted(df):
    output = BytesIO()
    df_cleaned = df[~df.index.duplicated(keep='first')]
    df_transposed = df_cleaned.T.reset_index()
    df_transposed.columns = ['Metrics'] + list(df_transposed.columns[1:])
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_transposed.to_excel(writer, index=False, sheet_name='Statistics', startrow=1)
        workbook = writer.book
        worksheet = writer.sheets['Statistics']
        header_format = workbook.add_format({'bold': True, 'bg_color': '#44723c', 'font_color': 'white', 'border': 1})
        cell_format = workbook.add_format({'border': 1, 'num_format': '0.00'})
        alt_row_format = workbook.add_format({'bg_color': '#f2f2f2', 'border': 1, 'num_format': '0.00'})
        text_format = workbook.add_format({'border': 1})
        
        for col_num, value in enumerate(df_transposed.columns.values):
            worksheet.write(0, col_num, value, header_format)
            worksheet.set_column(col_num, col_num, 26)

        for row_num in range(1, len(df_transposed) + 1):
            row_format = alt_row_format if row_num % 2 == 0 else cell_format
            for col_num in range(len(df_transposed.columns)):
                val = df_transposed.iat[row_num - 1, col_num]
                if isinstance(val, (float, int)) and not (pd.isna(val) or np.isinf(val)):
                    worksheet.write(row_num, col_num, val, row_format)
                else:
                    worksheet.write(row_num, col_num, str(val), text_format)
       
        worksheet.freeze_panes(1, 1)
    return output.getvalue()

# ---- Streamlit UI ----

st.title("Portfolio Statistics Tool")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    try:
        sheet = st.selectbox("Select Sheet", pd.ExcelFile(uploaded_file, engine='openpyxl').sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet, engine='openpyxl')
        df.set_index("date", inplace=True)
    except Exception as e:
        st.error(f"❌ Failed to read Excel file: {e}")
        st.stop()

    benchmarkColumn = st.selectbox("Select Benchmark Column", [col for col in df.columns if col != "signal"])

    if st.button("Generate Statistics"):
        try:
            stats_df = standardStats(df, benchmarkColumn)
            st.success("✅ Statistics generated successfully!")
            st.dataframe(stats_df)
            st.download_button(
                label="Download Results as Excel",
                data=to_excel_formatted(stats_df),
                file_name="formatted_financial_statistics.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"❌ Error: {e}")
