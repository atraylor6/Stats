import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from scipy.stats import skew
from fredapi import Fred

# -----------------------------
# Define All Helper Functions
# -----------------------------

signal = df[["signal"]]
returns = df.drop(columns=["signal"])

benchmarkColumn = "benchmark"

rolling_window_3yr = 36
rolling_window_5yr = 60

def annualizedReturn(returns):
    productReturns = (1 + returns).prod()
    totalRows = returns.shape[0]
    geometricMeanReturn = (productReturns ** (12 / totalRows)) - 1
    return geometricMeanReturn * 100

def annualizedReturnOther(returns):
    productReturns = (1 + returns).prod()
    totalRows = returns.shape[0]
    geometricMeanReturn = (productReturns ** (12 / totalRows))
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
        ) - data[benchmarkColumn].rolling(window=window_size).apply(
            lambda x: annualizedReturn(x)
        ).dropna()
    
    return rollingExcessReturns

def annualizedStandardDeviation(returns):
    monthlyStd = returns.std()
    annualStd = (monthlyStd * (12 ** 0.5)) * 100
    return annualStd

excessreturns = returns.copy()

for column in excessreturns.columns:
    if column != benchmarkColumn:
        excessreturns[column] = returns[column] - returns[benchmarkColumn]

excessreturns = excessreturns.drop(columns=[benchmarkColumn])

def downsideDeviation(excessreturns):
    downsideSquared = (excessreturns[excessreturns < 0] ** 2).sum()
    numValues = (excessreturns < 0).shape[0]
    dev = downsideSquared / numValues
    sqrtDev = np.sqrt(dev)
    annualizedSqrtDev = sqrtDev * np.sqrt(12)
    return annualizedSqrtDev * 100

def excessReturnSkew(excessreturns):
    skewness_values = excessreturns.apply(skew, nan_policy='omit')
    return skewness_values

def trackingError(excessreturns):
    tracking_error = (excessreturns.std() * np.sqrt(12)) * 100
    return tracking_error

def average10Yr(df, apiKey):
    fred = Fred(api_key=apiKey)
    startDate = df.index.min().strftime('%Y-%m-%d')
    endDate = df.index.max().strftime('%Y-%m-%d')
    seriesId = 'GS10'
    data = fred.get_series(seriesId, startDate, endDate)
    averageRate = data.mean()
    return averageRate

average_10year_rate = average10Yr(df, apiKey)

def sharpeRatio(returns, average10year):
    annualized_returns = returns.apply(annualizedReturn)
    excess_return = annualized_returns - average10year
    std_dev = annualizedStandardDeviation(returns)
    sharpe_ratio = excess_return / std_dev
    return sharpe_ratio

def informationRatio(returns, benchmarkColumn):
    excess_return = meanExcessReturn(returns, benchmarkColumn)
    te = trackingError(excessreturns)
    information_ratio = excess_return / te
    return information_ratio

def sortinoRatio(returns, benchmarkColumn):
    excess_return = meanExcessReturn(returns, benchmarkColumn)
    dd = downsideDeviation(excessreturns)
    sortino_ratio = excess_return / dd
    return sortino_ratio

def calculateBeta(returns, benchmarkColumn):
    cov_matrix = returns.cov()
    benchmark_var = cov_matrix[benchmarkColumn][benchmarkColumn]
    benchmark_cov = cov_matrix[benchmarkColumn]
    beta = benchmark_cov / benchmark_var
    return beta

def calculateAlpha(returns, benchmarkColumn, average10year, beta_values):
    annualized_returns = returns.apply(annualizedReturn)
    benchmark_annualized_return = annualized_returns[benchmarkColumn]
    alpha = annualized_returns - ((benchmark_annualized_return - average10year) * beta_values + average10year)
    return alpha

def correlation(df, benchmarkColumn):
    return df.corr()[benchmarkColumn]

def rSquared(df, benchmarkColumn):
    correlation_values = correlation(df, benchmarkColumn)
    r_squared_values = correlation_values ** 2
    return r_squared_values

def upside(returns, benchmarkColumn):
    upside_df = returns[returns[benchmarkColumn] > 0].copy()
    return upside_df

upside_df = upside(returns, benchmarkColumn)

def upsideCapture(upside, benchmarkColumn):
    annualized_returns = upside.apply(annualizedReturn)
    benchmark_annualized_return = annualized_returns[benchmarkColumn]
    upside_capture_ratios = annualized_returns / benchmark_annualized_return
    return upside_capture_ratios

def downside(returns, benchmarkColumn):
    downside_df = returns[returns[benchmarkColumn] < 0].copy()
    return downside_df

downside_df = downside(returns, benchmarkColumn)

def downsideCapture(downside, benchmarkColumn):
    annualized_returns = downside.apply(annualizedReturn)
    benchmark_annualized_return = annualized_returns[benchmarkColumn]
    downside_capture_ratios = annualized_returns / benchmark_annualized_return
    return downside_capture_ratios

def rolling12MOutUnderPerf(returns, benchmarkColumn, threshold=0.01):
    months = 12
    stats = {}

    for column in returns.columns:
        if column == benchmarkColumn:
            continue
        excess = (
            (1 + returns[column]).rolling(months).apply(np.prod, raw=True) -
            (1 + returns[benchmarkColumn]).rolling(months).apply(np.prod, raw=True)
        )

        total_windows = excess.count()
        outperform = (excess > threshold).sum()
        underperform = (excess < -threshold).sum()

        stats[column] = {
            "Pct_Outperform_>1%": (outperform / total_windows) * 100,
            "Pct_Underperform_>1%": (underperform / total_windows) * 100
        }

    return pd.DataFrame(stats).T

def relativeMaxDrawdown(returns, benchmarkColumn):
    relative_dd = {}

    for column in returns.columns:
        if column == benchmarkColumn:
            continue
        relative_returns = returns[column] - returns[benchmarkColumn]
        cumulative_relative = (1 + relative_returns).cumprod()
        peak = cumulative_relative.cummax()
        drawdown = (cumulative_relative - peak) / peak
        relative_dd[column] = drawdown.min() * 100  # in percent

    return pd.Series(relative_dd)

def standardStats(df, apiKey):
    average_10year_rate = average10Yr(df, apiKey)
    mean_excess_returns = meanExcessReturn(returns, benchmarkColumn)
    rolling_excess_returns_3yr = calculateRollingExcessReturns(returns, rolling_window_3yr, benchmarkColumn)
    rolling_excess_returns_5yr = calculateRollingExcessReturns(returns, rolling_window_5yr, benchmarkColumn)
    annualized_std_dev = annualizedStandardDeviation(returns)
    rolling_excess_returns_3yr_skewness = excessReturnSkew(rolling_excess_returns_3yr)
    te = trackingError(excessreturns)
    sharpe_ratios = sharpeRatio(returns, average_10year_rate)
    info_ratios = informationRatio(returns, benchmarkColumn)
    sortino_ratios = sortinoRatio(returns, benchmarkColumn)
    beta_values = calculateBeta(returns, benchmarkColumn)
    alphas = calculateAlpha(returns, benchmarkColumn, average_10year_rate, beta_values)
    correlation_values = correlation(returns, benchmarkColumn)
    r_squared_values = rSquared(returns, benchmarkColumn)
    upside_df = upside(returns, benchmarkColumn)
    upside_capture_ratios = upsideCapture(upside_df, benchmarkColumn)
    downside_df = downside(returns, benchmarkColumn)
    downside_capture_ratios = downsideCapture(downside_df, benchmarkColumn)

    # New metrics
    relative_max_drawdown = relativeMaxDrawdown(returns, benchmarkColumn)
    rolling_out_under_perf = rolling12MOutUnderPerf(returns, benchmarkColumn)

    stats_df = pd.DataFrame({
        "Annualized Return": annualizedReturn(returns),
        "Excess Return": mean_excess_returns,
        "Standard Deviation": annualized_std_dev,
        "Downside Deviation": downsideDeviation(excessreturns),
        "3-Year Rolling Excess Return Skewness": rolling_excess_returns_3yr_skewness,
        "Tracking Error": te,
        "Sharpe Ratio": sharpe_ratios,
        "Information Ratio": info_ratios,
        "Sortino Ratio": sortino_ratios,
        "Beta": beta_values,
        "Alpha": alphas,
        "R-Squared": r_squared_values,
        "Correlation": correlation_values,
        "Upside Capture": upside_capture_ratios,
        "Downside Capture": downside_capture_ratios,
        "Relative Max Drawdown vs Benchmark": relative_max_drawdown,
        "% Rolling 12M Outperformance > 1%": rolling_out_under_perf["Pct_Outperform_>1%"],
        "% Rolling 12M Underperformance < -1%": rolling_out_under_perf["Pct_Underperform_>1%"]
    })

    return stats

st.title("📊 Portfolio Statistics Tool")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

api_key = "d0fc5bc2297df338f8f31e08b197b1d9"

if uploaded_file:
    try:
        sheet = st.selectbox("Select Sheet", pd.ExcelFile(uploaded_file, engine='openpyxl').sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet, engine='openpyxl')
        df.set_index("date", inplace=True)

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
                    df_clean = df.replace([np.inf, -np.inf], np.nan)  # Clean up invalid values
                    df_clean = df_clean.fillna("")  # Fill NaNs to prevent Excel XML error
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_clean.to_excel(writer, index=True, sheet_name='Statistics')
                    output.seek(0)
                    return output.read()

                st.download_button(
                    label="Download Results as Excel",
                    data=to_excel(stats_df),
                    file_name="financial_statistics_table.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"❌ Error while generating stats: {e}")

    except Exception as e:
        st.error(f"❌ Failed to read Excel file: {e}")
