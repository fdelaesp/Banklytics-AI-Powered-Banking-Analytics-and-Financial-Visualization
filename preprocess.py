# src/preprocess.py
import pandas as pd
import numpy as np


def preprocess_sbp_data(filepath):
    """
    Reads the SBP Excel file, pivots the data, and calculates a suite of financial indicators.
    Returns a DataFrame with:
      [Bank, Year, Month, net_income, total_assets, equity, ROA, Leverage, ROE, classification,
       liquidity_ratio, deposit_diversity, deposit_view_to_plazo, coverage_ratio,
       leverage_ratio_extra, capitalization_ratio, adjusted_ROE]
    """
    # 1. Load data from Excel
    df = pd.read_excel(filepath)
    # Ensure 'Valor' is numeric
    df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce')

    # 2. Pivot the data: index by Subgrupo (Bank), Año, Mes; columns by (Categoría, Indicador)
    pivoted = df.pivot_table(
        index=["Subgrupo", "Año", "Mes"],
        columns=["Categoría", "Indicador"],
        values="Valor",
        aggfunc="sum"
    )

    # Helper function to safely get a column from the pivoted DataFrame
    def get_col(category, indicator):
        if (category, indicator) in pivoted.columns:
            return pivoted[(category, indicator)].fillna(0)
        else:
            return pd.Series(0, index=pivoted.index)

    # 3. Basic financial metrics
    net_income = get_col("Patrimonio", "Utilidad De Periodo")
    total_assets = get_col("Patrimonio", "Pasivo Y Patrimonio")
    capital = get_col("Patrimonio", "Capital")
    other_reserves = get_col("Patrimonio", "Otras Reservas")
    previous_net_income = get_col("Patrimonio", "Utilidad De Periodos Anteriores")
    market_value_gain = get_col("Patrimonio", "Ganancia O Perdida En Valores Disponible Para La Venta")

    equity = capital + other_reserves + previous_net_income + market_value_gain + net_income

    roa = net_income / total_assets.replace({0: np.nan})
    leverage = total_assets / equity.replace({0: np.nan})
    roe = roa * leverage

    # Classification based on ROE quantiles
    temp_roe = roe.dropna()
    if len(temp_roe) > 2:
        q33, q66 = temp_roe.quantile([0.33, 0.66])
    else:
        q33, q66 = 0, 0

    def classify_roe(x):
        if pd.isna(x):
            return "Unknown"
        elif x <= q33:
            return "Low performance"
        elif x <= q66:
            return "Medium performance"
        else:
            return "High performance"

    classification = roe.apply(classify_roe)

    # 4. Additional Financial Ratios
    # Liquidity Ratio = Total Activos Líquidos / Total Depositos
    if "Activos Liquidos" in pivoted.columns.get_level_values(0):
        total_activos_liquidos = pivoted["Activos Liquidos"].sum(axis=1)
    else:
        total_activos_liquidos = pd.Series(0, index=pivoted.index)

    if "Depositos" in pivoted.columns.get_level_values(0):
        total_deposits = pivoted["Depositos"].sum(axis=1)
    else:
        total_deposits = pd.Series(0, index=pivoted.index)

    liquidity_ratio = np.where(total_deposits != 0, total_activos_liquidos / total_deposits, np.nan)
    liquidity_ratio = pd.Series(liquidity_ratio, index=pivoted.index)

    # Deposits Diversity = Depositos De Particulares / Depositos De Bancos
    deposit_particulares = get_col("Depositos", "De Particulares")
    deposit_bancos = get_col("Depositos", "De Bancos")
    deposit_diversity = np.where(deposit_bancos != 0, deposit_particulares / deposit_bancos, np.nan)
    deposit_diversity = pd.Series(deposit_diversity, index=pivoted.index)

    # Deposit A La Vista to A Plazo Ratio = Depositos A La Vista / Depositos A Plazo
    deposit_a_la_vista = get_col("Depositos", "A La Vista")
    deposit_a_plazo = get_col("Depositos", "A Plazo")
    deposit_view_to_plazo = np.where(deposit_a_plazo != 0, deposit_a_la_vista / deposit_a_plazo, np.nan)
    deposit_view_to_plazo = pd.Series(deposit_view_to_plazo, index=pivoted.index)

    # Coverage Ratio = Cartera Crediticia Menos Provisiones / (Cartera Crediticia Locales + Cartera Crediticia Extranjero)
    credit_locales = get_col("Cartera Crediticia", "Locales")
    credit_extr = get_col("Cartera Crediticia", "Extranjero")
    credit_menos = get_col("Cartera Crediticia", "Menos Provisiones")
    credit_total = credit_locales + credit_extr
    coverage_ratio = np.where(credit_total != 0, credit_menos / credit_total, np.nan)
    coverage_ratio = pd.Series(coverage_ratio, index=pivoted.index)

    # Leverage Ratio (Additional) = (Obligaciones + Otros Pasivos) / Equity
    obligations_locales = get_col("Obligaciones", "Locales")
    obligations_extr = get_col("Obligaciones", "Extranjero")
    otros_pasivos_locales = get_col("Otros Pasivos", "Locales")
    otros_pasivos_extr = get_col("Otros Pasivos", "Extranjero")
    total_obligations = obligations_locales + obligations_extr + otros_pasivos_locales + otros_pasivos_extr
    leverage_ratio_extra = np.where(equity != 0, total_obligations / equity, np.nan)
    leverage_ratio_extra = pd.Series(leverage_ratio_extra, index=pivoted.index)

    # Capitalization Ratio = Equity / Total Assets
    capitalization_ratio = np.where(total_assets != 0, equity / total_assets, np.nan)
    capitalization_ratio = pd.Series(capitalization_ratio, index=pivoted.index)

    # Adjusted ROE: ROE adjusted by credit quality = ROE * (Cartera Crediticia Locales / (Cartera Crediticia Locales - Menos Provisiones Locales))
    credit_locales_minus = get_col("Cartera Crediticia", "Menos Provisiones Locales")
    denom_adjusted = credit_locales - credit_locales_minus
    adjusted_ROE = np.where(denom_adjusted != 0, roe * (credit_locales / denom_adjusted), np.nan)
    adjusted_ROE = pd.Series(adjusted_ROE, index=pivoted.index)

    # 5. Combine results into a DataFrame
    result = pd.DataFrame({
        "Bank": pivoted.index.get_level_values("Subgrupo"),
        "Year": pivoted.index.get_level_values("Año"),
        "Month": pivoted.index.get_level_values("Mes"),
        "net_income": net_income,
        "total_assets": total_assets,
        "equity": equity,
        "ROA": roa,
        "Leverage": leverage,
        "ROE": roe,
        "classification": classification,
        "liquidity_ratio": liquidity_ratio,
        "deposit_diversity": deposit_diversity,
        "deposit_view_to_plazo": deposit_view_to_plazo,
        "coverage_ratio": coverage_ratio,
        "leverage_ratio_extra": leverage_ratio_extra,
        "capitalization_ratio": capitalization_ratio,
        "adjusted_ROE": adjusted_ROE
    })

    # Reset index to get a flat DataFrame
    result = result.reset_index(drop=True)
    return result


if __name__ == "__main__":
    filepath = "data/SBP_Panama_Balance_de_Bancos.xlsx"
    processed_df = preprocess_sbp_data(filepath)
    processed_df.to_csv("data/financials_processed.csv", index=False)
    print("Processed data saved to 'data/financials_processed.csv'.")
