# app/dashboard.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# Cache model and data
@st.cache_resource
def load_model(model_path="models/decision_tree_model.joblib"):
    return joblib.load(model_path)

@st.cache_data
def load_processed_data(data_path="data/financials_processed.csv"):
    return pd.read_csv(data_path)

model = load_model()
data_df = load_processed_data()

# Sidebar: Language & Global Filters
st.sidebar.title("Settings")

language = st.sidebar.radio("Select Language / Seleccione el idioma", ["English", "Español"])

if language == "English":
    dashboard_title = "DuPont AI Analyzer – Financial Diagnostic"
    global_filter_title = "Global Filters"
    bank_filter_label = "Select Bank(s):"
    year_filter_label = "Select Year(s):"
    month_filter_label = "Select Month(s):"
    class_filter_label = "Select Classification(s):"
    menu_label = "Select Visualization"
    menu_options = [
        "3D Graph: ROA vs. Liquidity Ratio vs. Coverage Ratio",
        "3D Graph: Leverage vs. Capitalization vs. Adjusted ROE",
        "2D Graph: ROA vs. Liquidity Ratio",
        "2D Graph: ROA vs. Coverage Ratio",
        "2D Graph: Liquidity Ratio vs. Coverage Ratio",
        "Time Series: Average ROE Over Time",
        "Financial Metrics Table"
    ]
else:
    dashboard_title = "DuPont AI Analyzer – Diagnóstico Financiero"
    global_filter_title = "Filtros Globales"
    bank_filter_label = "Seleccione Banco(s):"
    year_filter_label = "Seleccione Año(s):"
    month_filter_label = "Seleccione Mes(es):"
    class_filter_label = "Seleccione Clasificación(es):"
    menu_label = "Seleccione Visualización"
    menu_options = [
        "Gráfico 3D: ROA vs. Ratio de Liquidez vs. Cobertura",
        "Gráfico 3D: Apalancamiento vs. Capitalización vs. ROE Ajustado",
        "Gráfico 2D: ROA vs. Ratio de Liquidez",
        "Gráfico 2D: ROA vs. Cobertura",
        "Gráfico 2D: Ratio de Liquidez vs. Cobertura",
        "Serie de Tiempo: ROE Promedio",
        "Tabla de Métricas Financieras"
    ]

st.title(dashboard_title)

# Sidebar: Global filters (applied to all visualizations)
st.sidebar.subheader(global_filter_title)
banks = sorted(data_df["Bank"].unique().tolist())
selected_banks = st.sidebar.multiselect(bank_filter_label, banks, default=banks)

years = sorted(data_df["Year"].unique().tolist())
selected_years = st.sidebar.multiselect(year_filter_label, years, default=years)

months = sorted(data_df["Month"].unique().tolist())
selected_months = st.sidebar.multiselect(month_filter_label, months, default=months)

classes = sorted(data_df["classification"].unique().tolist())
selected_classes = st.sidebar.multiselect(class_filter_label, classes, default=classes)

# Sidebar: Main visualization menu
visualization_option = st.sidebar.selectbox(menu_label, menu_options)

# Filter data based on global filters
filtered_df = data_df[
    data_df["Bank"].isin(selected_banks) &
    data_df["Year"].isin(selected_years) &
    data_df["Month"].isin(selected_months) &
    data_df["classification"].isin(selected_classes)
]

# Main area: Display the selected visualization
st.header(visualization_option)

if visualization_option in [
    menu_options[0], menu_options[1]
]:
    # 3D Graphs
    if filtered_df.empty:
        st.write("No data available for the selected filters.")
    else:
        if visualization_option in [menu_options[0], "Gráfico 3D: ROA vs. Ratio de Liquidez vs. Cobertura"]:
            # 3D Scatter: x=ROA, y=liquidity_ratio, z=coverage_ratio
            fig = px.scatter_3d(
                filtered_df,
                x="ROA",
                y="liquidity_ratio",
                z="coverage_ratio",
                color="Bank",
                symbol="classification",
                hover_data=["Bank", "Year", "Month", "ROE"],
                title=visualization_option
            )
        else:
            # 3D Scatter: x=Leverage, y=capitalization_ratio, z=adjusted_ROE
            fig = px.scatter_3d(
                filtered_df,
                x="Leverage",
                y="capitalization_ratio",
                z="adjusted_ROE",
                color="Bank",
                symbol="classification",
                hover_data=["Bank", "Year", "Month", "ROE"],
                title=visualization_option
            )
        st.plotly_chart(fig, use_container_width=True)

elif visualization_option in [
    menu_options[2], menu_options[3], menu_options[4]
]:
    # 2D Graphs
    if filtered_df.empty:
        st.write("No data available for the selected filters.")
    else:
        if visualization_option in [menu_options[2], "Gráfico 2D: ROA vs. Ratio de Liquidez"]:
            fig = px.scatter(
                filtered_df,
                x="ROA",
                y="liquidity_ratio",
                color="classification",
                hover_data=["Bank", "Year", "Month", "ROE"],
                title=visualization_option
            )
        elif visualization_option in [menu_options[3], "Gráfico 2D: ROA vs. Cobertura"]:
            fig = px.scatter(
                filtered_df,
                x="ROA",
                y="coverage_ratio",
                color="classification",
                hover_data=["Bank", "Year", "Month", "ROE"],
                title=visualization_option
            )
        else:
            fig = px.scatter(
                filtered_df,
                x="liquidity_ratio",
                y="coverage_ratio",
                color="classification",
                hover_data=["Bank", "Year", "Month", "ROE"],
                title=visualization_option
            )
        st.plotly_chart(fig, use_container_width=True)

elif visualization_option in [
    menu_options[5], "Serie de Tiempo: ROE Promedio", "Time Series: Average ROE Over Time"
]:
    # Time-Series Chart
    if filtered_df.empty:
        st.write("No data available for the selected filters.")
    else:
        df_time = filtered_df.copy()
        # Ensure Month is numeric; if not, map month names
        df_time["Month_num"] = pd.to_numeric(df_time["Month"], errors='coerce')
        if df_time["Month_num"].isna().all():
            month_map = {
                "January":1, "February":2, "March":3, "April":4, "May":5, "June":6,
                "July":7, "August":8, "September":9, "October":10, "November":11, "December":12,
                "Enero":1, "Febrero":2, "Marzo":3, "Abril":4, "Mayo":5, "Junio":6,
                "Julio":7, "Agosto":8, "Septiembre":9, "Octubre":10, "Noviembre":11, "Diciembre":12
            }
            df_time["Month_num"] = df_time["Month"].map(month_map)
        df_time = df_time.dropna(subset=["Month_num"])
        df_time["Month_num"] = df_time["Month_num"].astype(int)
        # Create a Date column
        df_time["Date"] = pd.to_datetime(df_time["Year"].astype(str) + "-" + df_time["Month_num"].astype(str) + "-01", format="%Y-%m-%d", errors='coerce')
        df_time = df_time.dropna(subset=["Date"])
        avg_roe_time = df_time.groupby("Date")["ROE"].mean().reset_index()
        fig_line = px.line(
            avg_roe_time,
            x="Date",
            y="ROE",
            title=visualization_option
        )
        st.plotly_chart(fig_line, use_container_width=True)

elif visualization_option in [menu_options[6], "Tabla de Métricas Financieras", "Financial Metrics Table"]:
    # Show the financial metrics table
    st.dataframe(filtered_df[["Bank", "Year", "Month", "liquidity_ratio", "deposit_diversity",
                              "deposit_view_to_plazo", "coverage_ratio", "leverage_ratio_extra",
                              "capitalization_ratio", "adjusted_ROE"]])
else:
    st.write("Select a visualization option from the sidebar.")

# --- Automatic Launch Wrapper for PyCharm ---
if __name__ == '__main__':
    import os, sys, subprocess
    if not os.getenv("STREAMLIT_RUN_ONCE"):
         os.environ["STREAMLIT_RUN_ONCE"] = "true"
         subprocess.run([sys.executable, "-m", "streamlit", "run", sys.argv[0], "--server.port=8501"])
