#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse av historiske CO2-utslippsintensiteter (2015-2024) med visualiseringer og beregninger.

Lager boxplott, medianprofiler, sammenligningsplott, JSD, og feilberegninger (RMSE, MAE, MBE).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import pkg_resources

# Konstanter
DATA_PATH = "/Users/pathujahvimalasenan/Desktop/Master/Data/CO2/Alle land/Alle_timer_2015-2024_CO2_aggregated.csv"
OUTPUT_DIR = "/Users/pathujahvimalasenan/Desktop/Master/Data/CO2/grafer/År/"
YEARS = list(range(2015, 2025))
MONTHS = ["jan", "feb", "mar", "apr", "mai", "jun", "jul", "aug", "sep", "okt", "nov", "des"]
PERIODS = {
    "2015-2017": list(range(2015, 2018)),
    "2018-2020": list(range(2018, 2021)),
    "2021-2024": list(range(2021, 2025)),
}
Y_LIMITS = {"2015-2017": (0, 400), "2018-2020": (0, 400), "2021-2024": (0, 400)}
FONT_SIZE = 22

# Ignorer irrelevante advarsler
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Sjekk pandas-versjon for kompatibilitet
pandas_version = pkg_resources.get_distribution("pandas").version
if pd.__version__ >= "2.0":
    print("Advarsel: Pandas 2.0+ oppdaget. Sørger for at Series konverteres til NumPy arrays for Matplotlib.")


def load_data(file_path):
    """Leser og validerer CSV-data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Filen {file_path} ble ikke funnet.")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Feil ved lesing av CSV-fil: {e}")

    def parse_datetime(row):
        try:
            return pd.Timestamp(int(row['År']), int(row['Måned']), int(row['Dag']), int(row['time']))
        except Exception as e:
            raise ValueError(f"Feil ved parsing av rad: {row}. Feil: {e}")

    df['Time'] = df.apply(parse_datetime, axis=1)
    if df['Time'].isna().any():
        raise ValueError("Ugyldige datetime-verdier i tidskolonnene.")
    if "Utslippsintensitet" not in df.columns:
        raise ValueError("Kolonne 'Utslippsintensitet' mangler i datafilen.")
    return df


def preprocess_data(df):
    """Forbereder data ved å filtrere og legge til kolonner."""
    df = df[df["Time"].dt.year != 2025].copy()
    df["År"] = df["Time"].dt.year
    df["Måned"] = df["Time"].dt.month
    df_no_2022 = df  # Ingen ekskludering av 2022 for utslippsdata
    return df, df_no_2022


def normalize_data(df_no_2022):
    """Normaliserer utslippsintensiteter til gjennomsnitt = 1 per år."""
    monthly_data = df_no_2022.groupby(["År", "Måned"])["Utslippsintensitet"].mean().reset_index()
    monthly_data["Normalisert_Utslipp"] = np.nan

    for year in YEARS:
        mask = monthly_data["År"] == year
        emissions = monthly_data.loc[mask, "Utslippsintensitet"].values
        if len(emissions) > 0 and np.mean(emissions) != 0:
            monthly_data.loc[mask, "Normalisert_Utslipp"] = emissions / np.mean(emissions)
    return monthly_data


def plot_boxplot(data, output_path):
    """Lager boxplott for normaliserte månedsutslipp."""
    fig, ax = plt.subplots(figsize=(15, 8))
    data_per_month = [
        data[data["Måned"] == m]["Normalisert_Utslipp"].dropna().values for m in range(1, 13)
    ]
    bp = ax.boxplot(
        data_per_month,
        positions=range(12),
        widths=0.6,
        patch_artist=True,
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", color="black", markersize=5),
    )
    for box in bp["boxes"]:
        box.set_facecolor("skyblue")

    ax.set_title(
        "Boxplott av normaliserte månedsutslipp (2015-2024, Gjennomsnitt = 1)",
        fontsize=FONT_SIZE,
    )
    ax.set_xlabel("Måned", fontsize=FONT_SIZE)
    ax.set_ylabel("Normalisert utslippsintensitet", fontsize=FONT_SIZE)
    ax.grid(True)
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS, rotation=45, ha="right", fontsize=FONT_SIZE - 8)
    ax.tick_params(axis="y", labelsize=FONT_SIZE)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_median_with_std(data, output_path):
    """Lager medianlinjeplott med ±1 standardavvik."""
    median_per_month = []
    std_per_month = []
    for m in range(1, 13):
        emissions = data[data["Måned"] == m]["Normalisert_Utslipp"].dropna().values
        median_per_month.append(np.median(emissions) if len(emissions) > 0 else np.nan)
        std_per_month.append(np.std(emissions, ddof=1) if len(emissions) > 0 else np.nan)

    upper_std = [m + s for m, s in zip(median_per_month, std_per_month)]
    lower_std = [m - s for m, s in zip(median_per_month, std_per_month)]

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(range(12), median_per_month, color="red", linestyle="--", linewidth=2, marker="o", label="Median")
    ax.fill_between(range(12), lower_std, upper_std, color="gray", alpha=0.3, label="±1 Standardavvik")

    ax.set_title("Syntetisk Årsprofil (2015-2024)", fontsize=FONT_SIZE)
    ax.set_xlabel("Måned", fontsize=FONT_SIZE)
    ax.set_ylabel("Normalisert utslippsintensitet", fontsize=FONT_SIZE)
    ax.grid(True)
    ax.legend(fontsize=FONT_SIZE - 4)
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS, rotation=45, ha="right", fontsize=FONT_SIZE - 8)
    ax.tick_params(axis="y", labelsize=FONT_SIZE)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Lagre median og std til CSV
    pd.DataFrame(
        {"Måned": range(1, 13), "Median": median_per_month, "Std": std_per_month}
    ).to_csv(os.path.join(OUTPUT_DIR, "median_std_per_måned_utslipp.csv"), index=False)
    return median_per_month, std_per_month


def plot_comparison(df, median_per_month, std_per_month, avg_emissions):
    """Lager sammenligningsplott for rådata vs. syntetisk profil."""
    for period, years in PERIODS.items():
        nrows = len(years)
        fig, axes = plt.subplots(nrows, 1, figsize=(15, 4 * nrows))
        axes = [axes] if nrows == 1 else axes.flatten()
        min_y, max_y = Y_LIMITS[period]

        for idx, year in enumerate(years):
            year_data = df[(df["År"] == year) & (df["Måned"].isin(range(1, 13)))].groupby("Måned")["Utslippsintensitet"].mean().reset_index()
            avg_emission = avg_emissions.get(year, 0)

            if avg_emission == 0:
                axes[idx].set_title(f"År {year} (ingen data)", fontsize=FONT_SIZE)
                axes[idx].grid(True)
                axes[idx].set_ylim(min_y, max_y)
                continue

            # Syntetisk profil
            synthetic = [median_per_month[m - 1] * avg_emission for m in range(1, 13)]
            synthetic_upper = [
                (median_per_month[m - 1] + std_per_month[m - 1]) * avg_emission for m in range(1, 13)
            ]
            synthetic_lower = [
                (median_per_month[m - 1] - std_per_month[m - 1]) * avg_emission for m in range(1, 13)
            ]

            # Plott rådata
            if not year_data.empty:
                x_data = year_data["Måned"].to_numpy()
                y_data = year_data["Utslippsintensitet"].to_numpy()
                axes[idx].plot(
                    x_data,
                    y_data,
                    label="Rådata",
                    color="blue",
                    linewidth=2,
                    marker="o",
                )

            # Plott syntetisk profil
            axes[idx].plot(
                range(1, 13),
                synthetic,
                label="Syntetisk Årsprofil",
                color="red",
                linestyle="--",
                linewidth=2,
                marker="o",
            )
            axes[idx].fill_between(range(1, 13), synthetic_lower, synthetic_upper, color="gray", alpha=0.3)

            axes[idx].set_title(f"År {year}", fontsize=FONT_SIZE)
            axes[idx].grid(True)
            axes[idx].set_ylim(min_y, max_y)
            axes[idx].set_xticks(range(1, 13))
            axes[idx].set_xticklabels(MONTHS, rotation=45, ha="right", fontsize=FONT_SIZE - 8)
            axes[idx].set_xlabel("Måned", fontsize=FONT_SIZE)
            axes[idx].tick_params(axis="y", labelsize=FONT_SIZE)

            if idx == nrows - 1:
                axes[idx].legend(loc="upper right", fontsize=FONT_SIZE - 4)

        fig.suptitle(f"Sammenligning Rådata vs. Syntetisk Profil ({period})", fontsize=FONT_SIZE + 2)
        fig.text(0.04, 0.5, "Utslippsintensitet (g CO₂/kWh)", va="center", rotation="vertical", fontsize=FONT_SIZE)
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
        plt.savefig(os.path.join(OUTPUT_DIR, f"sammenligning_syntetisk_rådata_{period}_utslipp.png"), dpi=300, bbox_inches="tight")
        plt.close()


def calculate_jsd(df, median_per_month, avg_emissions):
    """Beregner Jensen-Shannon Divergens per år."""
    def jensen_shannon_divergence(p, q):
        p = np.array(p) / (np.sum(p) + 1e-10)
        q = np.array(q) / (np.sum(q) + 1e-10)
        m = 0.5 * (p + q)
        return 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)

    jsd_values = {}
    scaler = MinMaxScaler()

    for year in YEARS:
        year_data = df[df["År"] == year].groupby("Måned")["Utslippsintensitet"].mean().reindex(range(1, 13)).values
        avg_emission = avg_emissions.get(year, 0)

        if avg_emission == 0 or np.all(np.isnan(year_data)):
            jsd_values[year] = np.nan
            continue

        synthetic = [median_per_month[m - 1] * avg_emission for m in range(1, 13)]
        valid_mask = ~(np.isnan(year_data) | np.isnan(synthetic))
        raw_valid = year_data[valid_mask]
        synth_valid = np.array(synthetic)[valid_mask]

        if len(raw_valid) > 0 and len(synth_valid) > 0:
            combined = np.concatenate([raw_valid, synth_valid]).reshape(-1, 1)
            normalized = scaler.fit_transform(combined).flatten()
            raw_norm = normalized[:len(raw_valid)]
            synth_norm = normalized[len(raw_valid):]
            jsd_values[year] = jensen_shannon_divergence(raw_norm, synth_norm)
        else:
            jsd_values[year] = np.nan

    # Stolpediagram
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(YEARS, [jsd_values.get(year, np.nan) for year in YEARS], color="skyblue", width=0.6)
    ax.set_title("Jensen-Shannon Divergens (2015-2024)", fontsize=FONT_SIZE)
    ax.set_xlabel("År", fontsize=FONT_SIZE)
    ax.set_ylabel("JSD", fontsize=FONT_SIZE)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_xticks(YEARS)
    ax.tick_params(axis="both", labelsize=FONT_SIZE - 2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "jsd_gjennomsnitt_per_år_utslipp.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Lagre JSD til CSV
    pd.DataFrame(
        {"År": YEARS, "JSD": [round(jsd_values.get(year, np.nan), 3) for year in YEARS]}
    ).to_csv(os.path.join(OUTPUT_DIR, "jsd_verdier_per_år_utslipp.csv"), index=False)
    return jsd_values


def calculate_errors(df, median_per_month, avg_emissions):
    """Beregner RMSE, MAE og MBE per år."""
    def mean_bias_error(y_true, y_pred):
        return np.mean(y_true - y_pred)

    errors = {"År": [], "RMSE": [], "MAE": [], "MBE": []}
    for year in YEARS:
        year_data = df[df["År"] == year].groupby("Måned")["Utslippsintensitet"].mean().reindex(range(1, 13)).values
        avg_emission = avg_emissions.get(year, 0)

        if avg_emission == 0 or np.all(np.isnan(year_data)):
            errors["År"].append(year)
            errors["RMSE"].append(np.nan)
            errors["MAE"].append(np.nan)
            errors["MBE"].append(np.nan)
            continue

        synthetic = [median_per_month[m - 1] * avg_emission for m in range(1, 13)]
        valid_mask = ~(np.isnan(year_data) | np.isnan(synthetic))
        raw_valid = year_data[valid_mask]
        synth_valid = np.array(synthetic)[valid_mask]

        if len(raw_valid) > 0 and len(synth_valid) > 0:
            rmse = np.sqrt(mean_squared_error(raw_valid, synth_valid))
            mae = mean_absolute_error(raw_valid, synth_valid)
            mbe = mean_bias_error(raw_valid, synth_valid)
            errors["År"].append(year)
            errors["RMSE"].append(round(rmse, 3))
            errors["MAE"].append(round(mae, 3))
            errors["MBE"].append(round(mbe, 3))
        else:
            errors["År"].append(year)
            errors["RMSE"].append(np.nan)
            errors["MAE"].append(np.nan)
            errors["MBE"].append(np.nan)

    pd.DataFrame(errors).to_csv(os.path.join(OUTPUT_DIR, "rmse_mae_mbe_per_år_utslipp.csv"), index=False)
    return errors


def main():
    """Hovedfunksjon for å kjøre analysen."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    df, df_no_2022 = preprocess_data(df)

    # Beregn årlig gjennomsnittsutslipp
    avg_emissions = df.groupby("År")["Utslippsintensitet"].mean().to_dict()
    print("Årlig gjennomsnittsutslipp (g CO₂/kWh):")
    print(pd.DataFrame({"År": avg_emissions.keys(), "Gjennomsnittsutslipp": avg_emissions.values()}))

    # Normaliser data
    monthly_data = normalize_data(df_no_2022)

    # Generer plott
    plot_boxplot(monthly_data, os.path.join(OUTPUT_DIR, "boxplot_gjennomsnitt_1_utslipp.png"))
    median_per_month, std_per_month = plot_median_with_std(
        monthly_data, os.path.join(OUTPUT_DIR, "median_med_std_gjennomsnitt_1_utslipp.png")
    )
    plot_comparison(df, median_per_month, std_per_month, avg_emissions)
    calculate_jsd(df, median_per_month, avg_emissions)
    calculate_errors(df, median_per_month, avg_emissions)

    # Skriv ut bekreftelse
    print("Alle plott og tabeller er lagret i:", OUTPUT_DIR)
    print(f"1. Boxplott: boxplot_gjennomsnitt_1_utslipp.png")
    print(f"2. Median med std: median_med_std_gjennomsnitt_1_utslipp.png")
    print("3-5. Sammenligningsplott for perioder:")
    for period in PERIODS:
        print(f"   - {period}: sammenligning_syntetisk_rådata_{period}_utslipp.png")
    print(f"JSD-plott: jsd_gjennomsnitt_per_år_utslipp.png")
    print(f"Tabeller: median_std_per_måned_utslipp.csv, jsd_verdier_per_år_utslipp.csv, rmse_mae_mbe_per_år_utslipp.csv")


if __name__ == "__main__":
    main()