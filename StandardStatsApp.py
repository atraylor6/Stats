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
    numValues = (excessreturns < 0).shape[0]
    dev = downsideSquared / numValues
    sqrtDev = np.sqrt(dev)
    annualizedSqrtDev = sqrtDev * np.sqrt(12)
    return annualizedSqrtDev * 100

def excessReturnSkew(excessreturns):
    skewness_values = excessreturns.apply(skew, nan_policy='omit')
    return skewness_values

def trackingError(excessreturns):
    return (excessreturns.std() * np.sqrt(12)) * 100

def average10Yr(df, apiKey):
    try:
        from fredapi import Fred
        fred = Fred(api_key=apiKey)
        startDate = df.index.min().strftime('%Y-%m-%d')
        endDate = df.index.max().strftime('%Y-%m-%d')
        data = fred.get_series('GS10', startDate, endDate)
        return data.mean()
    except Exception as e:
        return 3.5  # fallback annual yield in %


def sharpeRatio(returns, avg10yr):
    ann_return = returns.apply(annualizedReturn)
    excess = ann_return - avg10yr
    std_dev = annualizedStandardDeviation(returns)
    return excess / std_dev

def informationRatio(returns, benchmarkColumn):
    excess_return = meanExcessReturn(returns, benchmarkColumn)
    te = trackingError(returns.drop(columns=[benchmarkColumn]))
    return excess_return / te

def sortinoRatio(returns, benchmarkColumn):
    excess_return = meanExcessReturn(returns, benchmarkColumn)
    dd = downsideDeviation(returns.drop(columns=[benchmarkColumn]))
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
