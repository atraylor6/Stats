
def average10Yr_dummy():
    return 3.5  # You can later replace with FRED API

# ---- Stat functions ----

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from scipy.stats import skew
from fredapi import Fred

# -----------------------------
# Define All Helper Functions
# -----------------------------

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

def calculateRollingExcessReturns(data, window_size, benchmark_column):
    rollingExcessReturns = pd.DataFrame()
    for column in data.columns:
        rollingExcessReturns[f"{column}"] = (
            data[column].rolling(window=window_size).apply(
                lambda x: annualizedReturn(x)
            ).dropna()
        ) - data[benchmark_column].rolling(window=window_size).apply(
            lambda x: annualizedReturn(x)
        ).dropna()
    return rollingExcessReturns

def annualizedStandardDeviation(returns):
    monthlyStd = returns.std()
    annualStd = (monthlyStd * (12 ** 0.5)) * 100
    return annualStd

def downsideDeviation(excessreturns):
    downsideSquared = (excessreturns[excessreturns < 0] ** 2).sum()
    numValues = (excessreturns < 0).sum()
    dev = downsideSquared / numValues
    sqrtDev = np.sqrt(dev)
    return sqrtDev * np.sqrt(12) * 100

def excessReturnSkew(excessreturns):
    skewness_values = excessreturns.apply(skew, nan_policy='omit')
    return skewness_values

def trackingError(excessreturns):
    return (excessreturns.std() * np.sqrt(12)) * 100

def average10Yr():
    return 3.5  # You can later replace with FRED API

def sharpeRatio(returns, avg10yr):
    ann_return = returns.apply(annualizedReturn)
    excess = ann_return - avg10yr
    std_dev = annualizedStandardDeviation(returns)
    return excess / std_dev

def informationRatio(excess_return, excessreturns):
    te = trackingError(excessreturns)
    return excess_return / te

def sortinoRatio(excess_return, excessreturns):
    dd = downsideDeviation(excessreturns)
    return excess_return / dd

def calculateBeta(returns, benchmarkColumn):
    cov_matrix = returns.cov()
    benchmark_var = cov_matrix[benchmarkColumn][benchmarkColumn]
    benchmark_cov = cov_matrix[benchmarkColumn]
    return benchmark_cov / benchmark_var

def calculateAlpha(returns, benchmarkColumn, avg10yr, beta):
    ann_returns = returns.apply(annualizedReturn)
    bench_ann_return = ann_returns[benchmarkColumn]
    return ann_returns - ((bench_ann_return - avg10yr) * beta + avg10yr)

def correlation(df, benchmarkColumn):
    return df.corr()[benchmarkColumn]

def rSquared(df, benchmarkColumn):
    return correlation(df, benchmarkColumn) ** 2

def upside(returns, benchmarkColumn):
    return returns[returns[benchmarkColumn] > 0]

def upsideCapture(upside, benchmarkColumn):
    ann = upside.apply(annualizedReturn)
    return ann / ann[benchmarkColumn]

def downside(returns, benchmarkColumn):
    return returns[returns[benchmarkColumn] < 0]

def downsideCapture(downside, benchmarkColumn):
    ann = downside.apply(annualizedReturn)
    return ann / ann[benchmarkColumn]

def relativeMaxDrawdown(returns, benchmarkColumn):
    relative_dd = {}
    for column in returns.columns:
        if column == benchmarkColumn:
            continue
        relative = returns[column] - returns[benchmarkColumn]
        cum_rel = (1 + relative).cumprod()
        peak = cum_rel.cummax()
        drawdown = (cum_rel - peak) / peak
        relative_dd[column] = drawdown.min() * 100
    return pd.Series(relative_dd)

def rolling12MOutUnderPerf(returns, benchmarkColumn, threshold=0.01):
    months = 12
    stats = {}
    for col in returns.columns:
        if col == benchmarkColumn:
            continue
        excess = (
            (1 + returns[col]).rolling(months).apply(np.prod) -
            (1 + returns[benchmarkColumn]).rolling(months).apply(np.prod)
        )
        total = excess.count()
        stats[col] = {
            'Pct_Outperform_>1%': (excess > threshold).sum() / total * 100,
            'Pct_Underperform_<-1%': (excess < -threshold).sum() / total * 100
        }
    return pd.DataFrame(stats).T

def standardStats(df, apiKey, benchmarkColumn):
    returns = df.drop(columns=["signal"]) if "signal" in df.columns else df.copy()
    returns.index = pd.to_datetime(returns.index)
    excessreturns = returns.drop(columns=[benchmarkColumn]).subtract(returns[benchmarkColumn], axis=0)

    average_10year_rate = average10Yr(df, apiKey)
    mean_excess_returns = meanExcessReturn(returns, benchmarkColumn)
    ann_std_dev = annualizedStandardDeviation(returns)
    skewness = excessReturnSkew(excessreturns)
    te = trackingError(excessreturns)
    sharpe = sharpeRatio(returns, average_10year_rate)
    info = informationRatio(mean_excess_returns, excessreturns)
    sortino = sortinoRatio(mean_excess_returns, excessreturns)
    beta = calculateBeta(returns, benchmarkColumn)
    alpha = calculateAlpha(returns, benchmarkColumn, average_10year_rate, beta)
    r2 = rSquared(returns, benchmarkColumn)
    corr = correlation(returns, benchmarkColumn)
    up_df = upside(returns, benchmarkColumn)
    down_df = downside(returns, benchmarkColumn)
    up_capture = upsideCapture(up_df, benchmarkColumn)
    down_capture = downsideCapture(down_df, benchmarkColumn)
    rel_dd = relativeMaxDrawdown(returns, benchmarkColumn)
    rolling_stats = rolling12MOutUnderPerf(returns, benchmarkColumn)

    stats = pd.DataFrame({
        "Annualized Return": annualizedReturn(returns),
        "Excess Return": mean_excess_returns,
        "Standard Deviation": ann_std_dev,
        "Downside Deviation": downsideDeviation(excessreturns),
        "3-Year Excess Return Skewness": skewness,
        "Tracking Error": te,
        "Sharpe Ratio": sharpe,
        "Information Ratio": info,
        "Sortino Ratio": sortino,
        "Beta": beta,
        "Alpha": alpha,
        "R-Squared": r2,
        "Correlation": corr,
        "Upside Capture": up_capture,
        "Downside Capture": down_capture,
        "Relative Max Drawdown vs Benchmark": rel_dd,
        "% Rolling 12M Outperformance > 1%": rolling_stats['Pct_Outperform_>1%'],
        "% Rolling 12M Underperformance < -1%": rolling_stats['Pct_Underperform_<-1%']
    })

    return stats

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("📊 Portfolio Statistics Tool")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

api_key = "d0fc5bc2297df338f8f31e08b197b1d9"

if uploaded_file:
    try:
        sheet = st.selectbox("Select Sheet", pd.ExcelFile(uploaded_file, engine='openpyxl').sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet, engine='openpyxl')
        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index)

        st.write("🧪 DataFrame Preview", df.head())

        benchmarkColumn = st.selectbox("Select Benchmark Column", [col for col in df.columns if col != "signal"])

        if st.button("Generate Statistics"):
            try:
                st.write("📊 Benchmark column selected:", benchmarkColumn)
                stats_df = standardStats(df, api_key, benchmarkColumn)
                st.success("✅ Statistics generated successfully!")
                st.dataframe(stats_df)

                def to_excel(df):
                    output = BytesIO()
                    df_clean = df.replace([np.inf, -np.inf], np.nan).fillna("")
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_clean.to_excel(writer, index=True, sheet_name='Statistics')
                    output.seek(0)
                    return output.read()

                st.download_button(
                    label="📥 Download Results as Excel",
                    data=to_excel(stats_df),
                    file_name="financial_statistics_table.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"❌ Error while generating stats: {e}")
    except Exception as e:
        st.error(f"❌ Failed to read Excel file: {e}")
