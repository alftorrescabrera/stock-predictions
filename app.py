"""Streamlit app for stock prediction project."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.baseline_models import BaselineModels
from src.config import (
    DATA_DIR,
    DISABLE_HF_XET,
    EMBEDDING_MODEL_NAME,
    LOCAL_FILES_ONLY,
    NEWS_CACHE_FILE,
    PRICE_FILE,
    PROCESSED_DATA_FILE,
    SENTIMENT_MODEL_NAME,
)
from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.evaluator import Evaluator
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.news_processor import NewsProcessor
from src.facade import StockPredictionFacade

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Stock Prediction", layout="wide")


def build_facade() -> StockPredictionFacade:
    """Create a facade configured with project defaults."""
    data_loader = DataLoader(PRICE_FILE, DATA_DIR / "news.csv")
    data_validator = DataValidator()
    news_processor = NewsProcessor(
        embedding_model_name=EMBEDDING_MODEL_NAME,
        sentiment_model_name=SENTIMENT_MODEL_NAME,
        cache_path=NEWS_CACHE_FILE,
        local_files_only=LOCAL_FILES_ONLY,
        disable_hf_xet=DISABLE_HF_XET,
    )
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    evaluator = Evaluator()
    baseline_models = BaselineModels()

    return StockPredictionFacade(
        data_loader=data_loader,
        data_validator=data_validator,
        news_processor=news_processor,
        feature_engineer=feature_engineer,
        model_trainer=model_trainer,
        evaluator=evaluator,
        baseline_models=baseline_models,
    )


@st.cache_resource
def get_facade() -> StockPredictionFacade:
    """Cache and return the configured facade instance."""
    return build_facade()


facade = get_facade()


def _baseline_table(baseline_metrics, task: str) -> pd.DataFrame:
    """Format baseline metrics for Streamlit display."""
    rows = []
    for name, metrics in baseline_metrics.get(task, {}).items():
        if task == "classification":
            rows.append(
                {
                    "model": name,
                    "accuracy": metrics.get("accuracy"),
                    "f1": metrics.get("f1"),
                    "roc_auc": metrics.get("roc_auc"),
                }
            )
        else:
            rows.append(
                {
                    "model": name,
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "r2": metrics.get("r2"),
                }
            )
    return pd.DataFrame(rows)

st.title("Stock Prediction")

st.header("Project Overview")
st.write(
    "This model predicts next-day stock direction (UP/DOWN) and next-day return using historical price data "
    "and market news. It combines price features with local news embeddings and FinBERT sentiment, and uses "
    "a chronological split to avoid leakage."
)
st.caption("Use the sections below to inspect data, build features, train models, and generate forecasts.")

st.header("Data Preview")
st.write("Load raw price and news data to verify inputs, date ranges, and tickers.")
if st.button("Load Data"):
    facade.load_data()

if facade.price_df is not None:
    st.subheader("Price Data")
    price_rows = st.slider("Price preview rows", min_value=5, max_value=200, value=20, step=5)
    price_tickers = sorted(facade.price_df["ticker"].unique())
    price_filter_tickers = st.multiselect("Filter price tickers", price_tickers, default=price_tickers[: min(3, len(price_tickers))])
    price_min_date = facade.price_df["date"].min().date()
    price_max_date = facade.price_df["date"].max().date()
    price_date_range = st.date_input("Filter price date range", value=(price_min_date, price_max_date), min_value=price_min_date, max_value=price_max_date)
    price_filtered = facade.price_df.copy()
    if price_filter_tickers:
        price_filtered = price_filtered[price_filtered["ticker"].isin(price_filter_tickers)]
    if isinstance(price_date_range, tuple) and len(price_date_range) == 2:
        start_date = pd.to_datetime(price_date_range[0])
        end_date = pd.to_datetime(price_date_range[1])
        price_filtered = price_filtered[price_filtered["date"].between(start_date, end_date)]
    st.dataframe(price_filtered.head(price_rows))

if facade.news_df is not None:
    st.subheader("News Data")
    news_rows = st.slider("News preview rows", min_value=5, max_value=200, value=20, step=5)
    news_tickers = sorted(facade.news_df["ticker"].unique())
    news_filter_tickers = st.multiselect("Filter news tickers", news_tickers, default=news_tickers[: min(3, len(news_tickers))])
    news_min_date = facade.news_df["date"].min().date()
    news_max_date = facade.news_df["date"].max().date()
    news_date_range = st.date_input("Filter news date range", value=(news_min_date, news_max_date), min_value=news_min_date, max_value=news_max_date)
    news_filtered = facade.news_df.copy()
    if news_filter_tickers:
        news_filtered = news_filtered[news_filtered["ticker"].isin(news_filter_tickers)]
    if isinstance(news_date_range, tuple) and len(news_date_range) == 2:
        start_date = pd.to_datetime(news_date_range[0])
        end_date = pd.to_datetime(news_date_range[1])
        news_filtered = news_filtered[news_filtered["date"].between(start_date, end_date)]
    st.dataframe(news_filtered.head(news_rows))

if facade.price_df is not None and facade.news_df is not None:
    report = facade.validate_data()
    st.subheader("Validation Report")
    st.write("Quality checks for missing values, duplicates, and news coverage by ticker/date.")
    st.json(report)

st.header("EDA")
st.write("Quick exploratory charts for prices, volume, and news coverage over time.")
if facade.price_df is not None:
    eda_tickers = sorted(facade.price_df["ticker"].unique())
    eda_filter_tickers = st.multiselect(
        "EDA tickers",
        eda_tickers,
        default=eda_tickers[: min(3, len(eda_tickers))],
    )
    eda_min_date = facade.price_df["date"].min().date()
    eda_max_date = facade.price_df["date"].max().date()
    eda_date_range = st.date_input(
        "EDA date range",
        value=(eda_min_date, eda_max_date),
        min_value=eda_min_date,
        max_value=eda_max_date,
    )
    price_eda = facade.price_df.copy()
    if eda_filter_tickers:
        price_eda = price_eda[price_eda["ticker"].isin(eda_filter_tickers)]
    if isinstance(eda_date_range, tuple) and len(eda_date_range) == 2:
        start_date = pd.to_datetime(eda_date_range[0])
        end_date = pd.to_datetime(eda_date_range[1])
        price_eda = price_eda[price_eda["date"].between(start_date, end_date)]

    price_fig = px.line(price_eda, x="date", y="close", color="ticker", title="Close Price Over Time")
    st.plotly_chart(price_fig, use_container_width=True)

    volume_fig = px.line(price_eda, x="date", y="volume", color="ticker", title="Volume Over Time")
    st.plotly_chart(volume_fig, use_container_width=True)

if facade.news_df is not None:
    news_eda = facade.news_df.copy()
    news_min_date = news_eda["date"].min().date()
    news_max_date = news_eda["date"].max().date()
    news_date_range = st.date_input(
        "EDA news date range",
        value=(news_min_date, news_max_date),
        min_value=news_min_date,
        max_value=news_max_date,
    )
    if isinstance(news_date_range, tuple) and len(news_date_range) == 2:
        start_date = pd.to_datetime(news_date_range[0])
        end_date = pd.to_datetime(news_date_range[1])
        news_eda = news_eda[news_eda["date"].between(start_date, end_date)]

    news_per_day = news_eda.groupby("date").size().reset_index(name="count")
    news_fig = px.line(news_per_day, x="date", y="count", title="News Count Over Time")
    st.plotly_chart(news_fig, use_container_width=True)

st.header("Feature Engineering")
st.write("Generate price/news features and cached embeddings used by the models.")
if LOCAL_FILES_ONLY and not NEWS_CACHE_FILE.exists():
    st.warning(
        "Local-only mode is enabled but no cached embeddings were found. "
        "Disable local-only mode or generate embeddings with internet access."
    )
col_feat_1, col_feat_2 = st.columns(2)
with col_feat_1:
    if st.button("Prepare Features"):
        with st.spinner("Processing news and engineering features..."):
            try:
                facade.prepare_features()
            except RuntimeError as exc:
                st.error(str(exc))
with col_feat_2:
    if st.button("Recompute Embeddings (force)"):
        with st.spinner("Recomputing embeddings and features..."):
            try:
                facade.prepare_features(force_recompute_news=True)
            except RuntimeError as exc:
                st.error(str(exc))

if facade.processed_df is not None:
    st.subheader("Processed Dataset")
    st.write("Inspect the engineered dataset used for training and prediction.")
    proc_rows = st.slider("Processed preview rows", min_value=5, max_value=200, value=20, step=5)
    show_embedding_columns = st.checkbox("Show expanded embedding columns", value=False)
    proc_tickers = sorted(facade.processed_df["ticker"].unique())
    proc_filter_tickers = st.multiselect(
        "Filter processed tickers",
        proc_tickers,
        default=proc_tickers[: min(3, len(proc_tickers))],
    )
    proc_min_date = facade.processed_df["date"].min().date()
    proc_max_date = facade.processed_df["date"].max().date()
    proc_date_range = st.date_input(
        "Filter processed date range",
        value=(proc_min_date, proc_max_date),
        min_value=proc_min_date,
        max_value=proc_max_date,
    )
    processed_filtered = facade.processed_df.copy()
    if proc_filter_tickers:
        processed_filtered = processed_filtered[processed_filtered["ticker"].isin(proc_filter_tickers)]
    if isinstance(proc_date_range, tuple) and len(proc_date_range) == 2:
        start_date = pd.to_datetime(proc_date_range[0])
        end_date = pd.to_datetime(proc_date_range[1])
        processed_filtered = processed_filtered[processed_filtered["date"].between(start_date, end_date)]
    if not show_embedding_columns:
        emb_cols = [
            c
            for c in processed_filtered.columns
            if c.startswith("weighted_emb_") or c.startswith("latest_emb_")
        ]
        processed_display = processed_filtered.drop(columns=emb_cols)
    else:
        processed_display = processed_filtered
    tab_all, tab_core = st.tabs(["All Columns", "Raw + Embeddings"])
    with tab_all:
        st.dataframe(processed_display.head(proc_rows))
    with tab_core:
        base_cols = [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "return",
            "target_next_return",
            "target_direction",
            "embedding",
            "weighted_embedding",
        ]
        available_cols = [c for c in base_cols if c in processed_filtered.columns]
        st.dataframe(processed_filtered[available_cols].head(proc_rows))

    st.subheader("Processed Columns Guide")
    st.write("Definitions for key engineered features and targets.")
    columns_guide = [
        ("return", "Daily close-to-close return per ticker"),
        ("return_lag_1", "Return from the previous trading day"),
        ("return_lag_2", "Return from two trading days ago"),
        ("return_lag_3", "Return from three trading days ago"),
        ("ma_5", "5-day moving average of close"),
        ("ma_10", "10-day moving average of close"),
        ("ma_20", "20-day moving average of close"),
        ("volatility_5", "5-day rolling std of returns"),
        ("volatility_10", "10-day rolling std of returns"),
        ("volume_change", "Daily percent change in volume"),
        ("volume_ma_5", "5-day moving average of volume"),
        ("high_low_range", "High minus low"),
        ("close_open_body", "Close minus open"),
        ("body_ratio", "Body divided by high-low range"),
        ("close_to_high", "Close relative to high"),
        ("close_to_low", "Close relative to low"),
        ("day_of_week", "Day of week (0=Mon)"),
        ("news_count", "Number of news items per ticker/day"),
        ("positive_count", "Count of positive news items"),
        ("negative_count", "Count of negative news items"),
        ("neutral_count", "Count of neutral news items"),
        ("mean_sentiment_score", "Average sentiment confidence score"),
        ("max_sentiment_score", "Max sentiment confidence score"),
        ("mean_sentiment_numeric", "Mean sentiment label mapped to -1/0/1"),
        ("abs_mean_sentiment", "Average absolute sentiment label"),
        ("latest_sentiment_numeric", "Sentiment label of latest news item"),
        ("weighted_embedding", "Vector: weighted mean embedding for the day"),
        ("embedding", "Vector: embedding from the latest news item"),
        ("weighted_emb_*", "Weighted mean embedding components (model-ready)"),
        ("latest_emb_*", "Latest embedding components (model-ready)"),
        ("target_next_return", "Next-day return (regression target)"),
        ("target_direction", "Next-day direction (classification target)"),
    ]
    st.dataframe(pd.DataFrame(columns_guide, columns=["column", "meaning"]))

    st.subheader("Weighted Embedding Example")
    st.write("Shows how multiple news items are aggregated into one daily embedding.")
    st.markdown(
        """For a given ticker/day with multiple news items, the weighted embedding is:

$\\text{weighted\\_embedding} = \\frac{\\sum_i w_i \\cdot \\text{embedding}_i}{\\sum_i w_i}$

where $w_i$ is based on sentiment (neutral = 1.0, else $1 + \\text{score}$)."""
    )
    example_df = facade.processed_df[["ticker", "date", "news_count", "weighted_embedding"]].head(3)
    st.dataframe(example_df)
    st.caption("Days with `news_count = 0` use zero vectors for embeddings.")

st.header("Model Training")
st.write("Train the XGBoost classifier/regressor using a temporal split.")
if st.button("Run Full Pipeline"):
    with st.spinner("Training models..."):
        try:
            facade.run_full_pipeline()
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
    st.success("Training complete. Models saved to /models.")

if facade.processed_df is not None:
    df = facade.processed_df.dropna(subset=["target_next_return"]).copy()
    train_df, val_df, test_df = facade.train_validation_test_split(df)
    st.write("Train range:", train_df["date"].min(), "to", train_df["date"].max())
    st.write("Validation range:", val_df["date"].min(), "to", val_df["date"].max())
    st.write("Test range:", test_df["date"].min(), "to", test_df["date"].max())

st.header("Metrics")
st.write(
    "Classification metrics summarize UP/DOWN prediction quality; regression metrics summarize return error."
)
if facade.metrics:
    st.subheader("Classification Metrics")
    st.caption("Key fields: accuracy, precision, recall, f1, f1_macro, roc_auc, balanced_accuracy.")
    st.json(facade.metrics.get("classification", {}))

    st.subheader("Regression Metrics")
    st.caption("Key fields: mae, rmse, r2.")
    st.json(facade.metrics.get("regression", {}))

if facade.metrics_by_ticker:
    st.subheader("Per-Ticker Classification Metrics")
    st.write("Per-ticker breakdown to identify symbols where the model performs better or worse.")
    st.caption(
        "accuracy: overall hit rate; precision: of predicted UP, how many were UP; "
        "recall: of actual UP, how many were found; f1: balance of precision/recall; "
        "f1_macro: average F1 across classes; roc_auc: ranking quality; "
        "balanced_accuracy: average of class recalls; support: rows in test; "
        "support_pos: UP rows; support_neg: DOWN rows."
    )
    metrics_rows = []
    for ticker, metrics in facade.metrics_by_ticker.items():
        metrics_rows.append(
            {
                "ticker": ticker,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "f1_macro": metrics.get("f1_macro"),
                "roc_auc": metrics.get("roc_auc"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "support": metrics.get("support"),
                "support_pos": metrics.get("support_pos"),
                "support_neg": metrics.get("support_neg"),
            }
        )
    per_ticker_df = pd.DataFrame(metrics_rows)
    st.dataframe(per_ticker_df.sort_values("f1_macro", ascending=False))

if facade.baseline_metrics:
    st.subheader("Baseline Comparison")
    st.info("Baseline comparisons are disabled in this build.")

st.header("Forecast")
st.write("Generate a next-day forecast for a selected ticker.")
if facade.processed_df is not None:
    if facade.price_df is not None:
        pred_tickers = sorted(facade.price_df["ticker"].unique())
    else:
        pred_tickers = sorted(facade.processed_df["ticker"].unique())

    ticker = st.selectbox("Predict for ticker", pred_tickers, key="pred_ticker")
    if st.button("Predict Latest"):
        if ticker not in set(facade.processed_df["ticker"].unique()):
            st.error("Ticker not found in processed dataset. Run feature prep first.")
        else:
            result = facade.predict_latest(ticker)
            st.caption(
                "latest_close: last observed close; predicted_direction: 1=UP, 0=DOWN; "
                "probability_up: model confidence for UP; predicted_return: next-day return; "
                "predicted_close: close implied by predicted_return."
            )
            st.json(result)

    st.subheader("Close Prices (+1 Day)")
    st.write("Historical close in blue; forecasted next-day close in red dashed line.")
    if facade.classifier is not None and facade.regressor is not None:
        hist_df = facade.processed_df[facade.processed_df["ticker"] == ticker].copy()
        hist_df = hist_df.sort_values("date")
        if not hist_df.empty:
            forecast = facade.predict_latest(ticker)
            last_date = pd.to_datetime(hist_df["date"].iloc[-1])
            next_date = last_date + pd.Timedelta(days=1)
            last_close = float(hist_df["close"].iloc[-1])
            predicted_close = float(forecast["predicted_close"])

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=hist_df["date"],
                    y=hist_df["close"],
                    mode="lines",
                    name="Historical Close",
                    line=dict(color="#2b6cb0", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[last_date, next_date],
                    y=[last_close, predicted_close],
                    mode="lines+markers",
                    name="Forecast (+1 day)",
                    line=dict(color="#e53e3e", width=3, dash="dash"),
                    marker=dict(size=8),
                )
            )
            fig.update_layout(title="Grafico de Close Prices", xaxis_title="Date", yaxis_title="Close")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Red dashed segment shows the predicted next-day close.")

st.header("Methodology / Limitations")
st.write(
    "- Stock prediction is noisy and uncertain.\n"
    "- Sentiment does not guarantee market movement.\n"
    "- Embeddings capture semantic meaning.\n"
    "- Chronological split is used to reduce leakage.\n"
    "- News is shifted so only past information is used."
)

st.header("Upload New Data (Price + News)")
st.write("Use new CSVs to rebuild embeddings, features, and models from scratch.")
st.write(
    "Upload fresh price/news CSVs to recompute embeddings and retrain the pipeline from scratch. "
    "This overwrites cached embeddings, processed data, and models."
)
price_upload = st.file_uploader("Upload price.csv", type=["csv"], key="price_upload")
news_upload = st.file_uploader("Upload news.csv", type=["csv"], key="news_upload")
if st.button("Run Pipeline on Uploaded Data"):
    if price_upload is None or news_upload is None:
        st.error("Please upload both price.csv and news.csv.")
    else:
        price_df = pd.read_csv(price_upload)
        news_df = pd.read_csv(news_upload)
        required_price = {"date", "ticker", "open", "high", "low", "close", "volume"}
        required_news = {"datetime", "ticker", "headline", "summary"}
        if not required_price.issubset(price_df.columns):
            st.error("price.csv missing required columns.")
            st.stop()
        if not required_news.issubset(news_df.columns):
            st.error("news.csv missing required columns.")
            st.stop()

        price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
        news_df["datetime"] = pd.to_datetime(news_df["datetime"], errors="coerce")
        news_df["date"] = news_df["datetime"].dt.date
        news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")

        with st.spinner("Recomputing embeddings and retraining models..."):
            try:
                facade.run_full_pipeline_on_data(price_df, news_df)
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()
        st.success("Pipeline complete with uploaded data.")
