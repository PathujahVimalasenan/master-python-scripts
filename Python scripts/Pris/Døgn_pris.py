#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse av timebaserte strømpriser per døgn (2015-2024) med visualiseringer og beregninger.

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

# Konstanter
DATA_PATH = "/Users/pathujahvimalasenan/Desktop/Master/Data/Historiske priser NVE/timesdata_pris_2015-2025.xlsx"
OUTPUT_DIR = "/Users/pathujahvimalasenan/Desktop/Master/Grafer/Døgn/"
YEARS = list(range(2015, 2025))
MONTHS = ["jan", "feb", "mar", "apr", "mai", "jun", "jul", "aug", "sep", "okt", "nov", "des"]
SELECTED_MONTHS = {"jan": 1, "jul": 7, "okt": 10}
HOURS = list(map(str, range(24)))
FONT_SIZE = 22
Y_LIMITS_YEAR = {
    2015: (0, 33), 2016: (0, 50), 2017: (0, 40), 2018: (0, 63), 2019: (0, 70),
    2020: (0, 30), 2021: (0, 225), 2022: (0, 450), 2023: (0, 160), 2024: (0, 130)
}
Y_LIMITS_MONTH = {"jan": (0, 170), "jul": (0, 180), "okt": (0, 180)}
PLOT_PROPS = {
    "medianprops": dict(color="red", linewidth=2),
    "whiskerprops": dict(color="black"),
    "capprops": dict(color="black"),
    "flierprops": dict(marker="o", color="black", markersize=5, linestyle="none")
}

# Ignorer irrelevante advarsler
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_data(file_path):
    """Leser og validerer Excel-data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Filen {file_path} ble ikke funnet.")
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        raise Exception(f"Feil ved lesing av Excel-fil: {e}")

    df["Time"] = pd.to_datetime(df["Hour"], format="%d.%m.%Y %H:%M", errors="coerce")
    if df["Time"].isna().any():
        raise ValueError("Ugyldige datetime-verdier i 'Hour'-kolonnen.")
    return df


def preprocess_data(df):
    """Forbereder data ved å filtrere og legge til kolonner."""
    df = df[df["Time"].dt.year != 2025].copy()
    df["År"] = df["Time"].dt.year
    df["Måned"] = df["Time"].dt.month
    df["Dag_i_måneden"] = df["Time"].dt.day
    df["Time_i_døgnet"] = df["Time"].dt.hour

    df_no_2022 = df[df["År"] != 2022].copy()
    return df, df_no_2022


def normalize_data(df_no_2022):
    """Normaliserer timepriser til gjennomsnitt = 1 per måned og år."""
    hourly_data = df_no_2022.groupby(["År", "Måned", "Time_i_døgnet"])["SPOTNO1(Øre/kWh)"].mean().reset_index()
    hourly_data["Normalisert_Pris"] = np.nan

    for year in YEARS:
        if year == 2022:
            continue
        for month in range(1, 13):
            mask = (hourly_data["År"] == year) & (hourly_data["Måned"] == month)
            prices = hourly_data.loc[mask, "SPOTNO1(Øre/kWh)"].values
            if len(prices) > 0 and np.mean(prices) != 0:
                hourly_data.loc[mask, "Normalisert_Pris"] = prices / np.mean(prices)
    return hourly_data


def plot_boxplot(hourly_data, output_path):
    """Lager boxplott for normaliserte timepriser per måned."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 20), sharex=True, sharey=True)
    axes = axes.flatten()
    median_std_data = {month: {"Median": [], "Std": []} for month in MONTHS}

    for month in range(1, 13):
        month_data = hourly_data[hourly_data["Måned"] == month]
        data_per_hour = [month_data[month_data["Time_i_døgnet"] == h]["Normalisert_Pris"].dropna().values 
                         for h in range(24)]
        bp = axes[month-1].boxplot(data_per_hour, positions=range(24), widths=0.6, 
                                   patch_artist=True, **PLOT_PROPS)
        for box in bp["boxes"]:
            box.set_facecolor("skyblue")

        median_per_hour = [np.median(d) if len(d) > 0 else np.nan for d in data_per_hour]
        std_per_hour = [np.std(d, ddof=1) if len(d) > 0 else np.nan for d in data_per_hour]
        median_std_data[MONTHS[month-1]] = {"Median": median_per_hour, "Std": std_per_hour}

        axes[month-1].scatter(range(24), median_per_hour, color="red", marker="o", s=50)
        axes[month-1].axhline(y=1, color="black", linestyle="--", linewidth=1, label="Gjennomsnitt = 1")
        axes[month-1].set_title(MONTHS[month-1], fontsize=FONT_SIZE)
        axes[month-1].grid(True)
        axes[month-1].tick_params(axis="y", labelsize=FONT_SIZE)
        if month == 12:
            axes[month-1].legend(fontsize=FONT_SIZE)

    for ax in axes:
        ax.set_xticks(range(24))
        ax.set_xticklabels(HOURS, rotation=45, ha="right", fontsize=FONT_SIZE-8)

    fig.suptitle("Boxplott av normaliserte timepriser (2015-2024, uten 2022)", fontsize=FONT_SIZE)
    fig.text(0.5, 0.04, "Time i døgnet", ha="center", fontsize=FONT_SIZE)
    fig.text(0.02, 0.5, "Normalisert spotpris", va="center", rotation="vertical", fontsize=FONT_SIZE)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Lagre median og std til CSV
    tabell_data = {"Time": range(24)}
    for month in MONTHS:
        tabell_data[f"{month}_Median"] = [round(val, 3) if not np.isnan(val) else np.nan 
                                         for val in median_std_data[month]["Median"]]
        tabell_data[f"{month}_Std"] = [round(val, 3) if not np.isnan(val) else np.nan 
                                      for val in median_std_data[month]["Std"]]
    pd.DataFrame(tabell_data).to_csv(os.path.join(OUTPUT_DIR, "median_std_per_time_og_måned.csv"), index=False)
    return median_std_data


def plot_median_with_std(median_std_data, output_path):
    """Lager medianlinjeplott for hver måned med ±1 standardavvik."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 20), sharex=True, sharey=True)
    axes = axes.flatten()

    for month in range(1, 13):
        median_per_hour = median_std_data[MONTHS[month-1]]["Median"]
        std_per_hour = median_std_data[MONTHS[month-1]]["Std"]
        upper_std = [m + s if not (np.isnan(m) or np.isnan(s)) else np.nan for m, s in zip(median_per_hour, std_per_hour)]
        lower_std = [m - s if not (np.isnan(m) or np.isnan(s)) else np.nan for m, s in zip(median_per_hour, std_per_hour)]

        axes[month-1].plot(range(24), median_per_hour, color="red", linewidth=2, label="Median")
        axes[month-1].scatter(range(24), median_per_hour, color="red", marker="o", s=50)
        axes[month-1].fill_between(range(24), lower_std, upper_std, color="gray", alpha=0.3)
        axes[month-1].axhline(y=1, color="black", linestyle="--", linewidth=1)
        axes[month-1].set_title(MONTHS[month-1], fontsize=FONT_SIZE)
        axes[month-1].grid(True)
        axes[month-1].legend(fontsize=FONT_SIZE-4)
        axes[month-1].tick_params(axis="y", labelsize=FONT_SIZE)
        axes[month-1].set_ylim(0.5, 1.35)

    for ax in axes:
        ax.set_xticks(range(24))
        ax.set_xticklabels(HOURS, rotation=45, ha="right", fontsize=FONT_SIZE-8)

    fig.suptitle("Syntetisk Døgnprofil (2015-2024, uten 2022)", fontsize=FONT_SIZE)
    fig.text(0.5, 0.04, "Time i døgnet", ha="center", fontsize=FONT_SIZE)
    fig.text(0.02, 0.5, "Normalisert spotpris", va="center", rotation="vertical", fontsize=FONT_SIZE)
    plt.tight_layout(rect=[0.06, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_yearly_comparison(df, median_std_data, daily_avg):
    """Lager sammenligningsplott for rådata vs. syntetisk profil per år."""
    for year in YEARS:
        fig, axes = plt.subplots(4, 3, figsize=(15, 20), sharex=True)
        axes = axes.flatten()
        min_y, max_y = Y_LIMITS_YEAR[year]

        for month in range(1, 13):
            year_month_data = df[(df["År"] == year) & (df["Måned"] == month)].groupby("Time_i_døgnet")["SPOTNO1(Øre/kWh)"].mean().reset_index()
            median_per_hour = median_std_data[MONTHS[month-1]]["Median"]
            std_per_hour = median_std_data[MONTHS[month-1]]["Std"]
            avg_price = daily_avg.get((year, month), 0)

            if avg_price == 0:
                axes[month-1].set_title(f"{MONTHS[month-1]} {year} (ingen data)", fontsize=FONT_SIZE)
                axes[month-1].grid(True)
                axes[month-1].set_ylim(min_y, max_y)
                continue

            synthetic = [val * avg_price if not np.isnan(val) else np.nan for val in median_per_hour]
            synthetic_std = [std * avg_price if not np.isnan(std) else np.nan for std in std_per_hour]
            synthetic_upper = [s + std for s, std in zip(synthetic, synthetic_std)]
            synthetic_lower = [s - std for s, std in zip(synthetic, synthetic_std)]

            if not year_month_data.empty:
                axes[month-1].plot(year_month_data["Time_i_døgnet"], year_month_data["SPOTNO1(Øre/kWh)"], 
                                   label="Rådata", color="blue", linewidth=2, marker="o")

            axes[month-1].plot(range(24), synthetic, label="Syntetisk Median-døgn", color="red", linestyle="--", linewidth=2, marker="o")
            axes[month-1].fill_between(range(24), synthetic_lower, synthetic_upper, color="gray", alpha=0.3)
            axes[month-1].set_title(MONTHS[month-1], fontsize=FONT_SIZE)
            axes[month-1].grid(True)
            axes[month-1].set_ylim(min_y, max_y)
            axes[month-1].tick_params(axis="y", labelsize=FONT_SIZE)
            if month == 3:
                axes[month-1].legend(loc="lower right", bbox_to_anchor=(0.98, 0.02), fontsize=FONT_SIZE-4)

        for ax in axes:
            ax.set_xticks(range(24))
            ax.set_xticklabels(HOURS, rotation=45, ha="right", fontsize=FONT_SIZE-8)

        fig.suptitle(f"Sammenligning Syntetisk Døgnprofil for År {year}", fontsize=FONT_SIZE+2)
        fig.text(0.5, 0.04, "Time i døgnet", ha="center", fontsize=FONT_SIZE)
        fig.text(0.04, 0.5, "Pris (Øre/kWh)", va="center", rotation="vertical", fontsize=FONT_SIZE)
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
        plt.savefig(os.path.join(OUTPUT_DIR, f"sammenligning_syntetisk_rådata_med_std_år_{year}.png"), dpi=300, bbox_inches="tight")
        plt.close()


def plot_selected_months(df, median_std_data, daily_avg):
    """Lager sammenligningsplott for januar, juli og oktober."""
    for month_name, month_num in SELECTED_MONTHS.items():
        fig, axes = plt.subplots(5, 2, figsize=(15, 25), sharex=True, sharey=True)
        axes = axes.flatten()
        min_y, max_y = Y_LIMITS_MONTH[month_name]

        for idx, year in enumerate(YEARS):
            year_month_data = df[(df["År"] == year) & (df["Måned"] == month_num)].groupby("Time_i_døgnet")["SPOTNO1(Øre/kWh)"].mean().reset_index()
            median_per_hour = median_std_data[month_name]["Median"]
            std_per_hour = median_std_data[month_name]["Std"]
            avg_price = daily_avg.get((year, month_num), 0)

            if avg_price == 0:
                axes[idx].set_title(f"{year} (ingen data)", fontsize=FONT_SIZE)
                axes[idx].grid(True)
                axes[idx].set_ylim(min_y, max_y)
                continue

            synthetic = [val * avg_price if not np.isnan(val) else np.nan for val in median_per_hour]
            synthetic_std = [std * avg_price if not np.isnan(std) else np.nan for std in std_per_hour]
            synthetic_upper = [s + std for s, std in zip(synthetic, synthetic_std)]
            synthetic_lower = [s - std for s, std in zip(synthetic, synthetic_std)]

            if not year_month_data.empty:
                axes[idx].plot(year_month_data["Time_i_døgnet"], year_month_data["SPOTNO1(Øre/kWh)"], 
                               label="Rådata", color="blue", linewidth=2, marker="o")

            axes[idx].plot(range(24), synthetic, label="Syntetisk Døgnprofil", color="red", linestyle="--", linewidth=2, marker="o")
            axes[idx].fill_between(range(24), synthetic_lower, synthetic_upper, color="gray", alpha=0.3, label="±1 Standardavvik")
            axes[idx].set_title(f"{year}", fontsize=FONT_SIZE)
            axes[idx].grid(True)
            axes[idx].set_ylim(min_y, max_y)
            axes[idx].tick_params(axis="y", labelsize=FONT_SIZE)
            if idx == 0:
                axes[idx].legend(fontsize=FONT_SIZE-4)

        for ax in axes:
            ax.set_xticks(range(24))
            ax.set_xticklabels(HOURS, rotation=45, ha="right", fontsize=FONT_SIZE-8)

        fig.suptitle(f"Sammenligning Syntetisk Døgnprofil for {month_name.capitalize()} (2015-2024)", fontsize=FONT_SIZE)
        fig.text(0.5, 0.04, "Time i døgnet", ha="center", fontsize=FONT_SIZE)
        fig.text(0.02, 0.5, "Pris (Øre/kWh)", va="center", rotation="vertical", fontsize=FONT_SIZE)
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
        plt.savefig(os.path.join(OUTPUT_DIR, f"sammenligning_syntetisk_rådata_med_std_{month_name}_alle_år.png"), dpi=300, bbox_inches="tight")
        plt.close()


def calculate_jsd(df, median_std_data, daily_avg):
    """Beregner Jensen-Shannon Divergens per år og måned."""
    def jensen_shannon_divergence(p, q):
        p = np.array(p) / (np.sum(p) + 1e-10)
        q = np.array(q) / (np.sum(q) + 1e-10)
        m = 0.5 * (p + q)
        return 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)

    jsd_values = {year: {month: np.nan for month in range(1, 13)} for year in YEARS}
    scaler = MinMaxScaler()

    for year in YEARS:
        for month in range(1, 13):
            year_month_data = df[(df["År"] == year) & (df["Måned"] == month)].groupby("Time_i_døgnet")["SPOTNO1(Øre/kWh)"].mean().reindex(range(24)).values
            median_per_hour = median_std_data[MONTHS[month-1]]["Median"]
            avg_price = daily_avg.get((year, month), 0)

            if avg_price == 0 or np.all(np.isnan(year_month_data)):
                continue

            synthetic = [val * avg_price if not np.isnan(val) else np.nan for val in median_per_hour]
            valid_mask = ~(np.isnan(year_month_data) | np.isnan(synthetic))
            raw_valid = year_month_data[valid_mask]
            synth_valid = np.array(synthetic)[valid_mask]

            if len(raw_valid) > 0 and len(synth_valid) > 0:
                combined = np.concatenate([raw_valid, synth_valid]).reshape(-1, 1)
                normalized = scaler.fit_transform(combined).flatten()
                raw_norm = normalized[:len(raw_valid)]
                synth_norm = normalized[len(raw_valid):]
                jsd_values[year][month] = jensen_shannon_divergence(raw_norm, synth_norm)

    # Stolpediagram for alle måneder
    fig, axes = plt.subplots(5, 2, figsize=(15, 25), sharey=True)
    axes = axes.flatten()
    for idx, year in enumerate(YEARS):
        jsd_per_month = [jsd_values[year][m] for m in range(1, 13)]
        axes[idx].bar(range(1, 13), jsd_per_month, color="skyblue")
        axes[idx].set_title(f"År {year}", fontsize=FONT_SIZE)
        axes[idx].grid(True, axis="y", linestyle="--", alpha=0.7)
        axes[idx].tick_params(axis="both", labelsize=FONT_SIZE)
        axes[idx].set_xticks(range(1, 13))
        axes[idx].set_xticklabels([m[:3] for m in MONTHS], rotation=45, ha="right", fontsize=FONT_SIZE-2)

    fig.suptitle("Jensen-Shannon Divergens per År (2015-2024)", fontsize=FONT_SIZE)
    fig.text(0.5, 0.04, "Måned", ha="center", fontsize=FONT_SIZE)
    fig.text(0.04, 0.5, "JSD-verdi", va="center", rotation="vertical", fontsize=FONT_SIZE)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, "jsd_sammenligning_per_år.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Stolpediagram for januar, juli, oktober
    fig, axes = plt.subplots(5, 2, figsize=(15, 25), sharey=True)
    axes = axes.flatten()
    for idx, year in enumerate(YEARS):
        jsd_per_month = [jsd_values[year][m] for m in SELECTED_MONTHS.values()]
        axes[idx].bar(SELECTED_MONTHS.keys(), jsd_per_month, color="skyblue")
        axes[idx].set_title(f"År {year}", fontsize=FONT_SIZE)
        axes[idx].grid(True, axis="y", linestyle="--", alpha=0.7)
        axes[idx].tick_params(axis="both", labelsize=FONT_SIZE)
        axes[idx].set_xticks(range(len(SELECTED_MONTHS)))
        axes[idx].set_xticklabels(SELECTED_MONTHS.keys(), rotation=45, ha="right", fontsize=FONT_SIZE-2)

    fig.suptitle("Jensen-Shannon Divergens for Januar, Juli, Oktober (2015-2024)", fontsize=FONT_SIZE)
    fig.text(0.5, 0.04, "Måned", ha="center", fontsize=FONT_SIZE)
    fig.text(0.04, 0.5, "JSD", va="center", rotation="vertical", fontsize=FONT_SIZE)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, "jsd_sammenligning_jan_jul_okt_per_år.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Lagre JSD til CSV
    jsd_data = {"År": YEARS}
    for month in range(1, 13):
        jsd_data[MONTHS[month-1]] = [round(jsd_values[year][month], 3) if not np.isnan(jsd_values[year][month]) else np.nan 
                                     for year in YEARS]
    pd.DataFrame(jsd_data).to_csv(os.path.join(OUTPUT_DIR, "jsd_verdier_per_år.csv"), index=False)

    # Lagre JSD for januar, juli, oktober
    jsd_selected_data = {"År": YEARS}
    for month_name, month_num in SELECTED_MONTHS.items():
        jsd_selected_data[month_name] = [round(jsd_values[year][month_num], 3) if not np.isnan(jsd_values[year][month_num]) else np.nan 
                                         for year in YEARS]
    pd.DataFrame(jsd_selected_data).to_csv(os.path.join(OUTPUT_DIR, "jsd_verdier_jan_jul_okt_per_år.csv"), index=False)


def calculate_errors(df, median_std_data, daily_avg):
    """Beregner RMSE, MAE og MBE per år og måned."""
    def mean_bias_error(y_true, y_pred):
        return np.mean(y_true - y_pred)

    errors = {"År": [], "Måned": [], "RMSE": [], "MAE": [], "MBE": []}
    for year in YEARS:
        for month in range(1, 13):
            year_month_data = df[(df["År"] == year) & (df["Måned"] == month)].groupby("Time_i_døgnet")["SPOTNO1(Øre/kWh)"].mean().reindex(range(24)).values
            median_per_hour = median_std_data[MONTHS[month-1]]["Median"]
            avg_price = daily_avg.get((year, month), 0)

            if avg_price == 0 or np.all(np.isnan(year_month_data)):
                errors["År"].append(year)
                errors["Måned"].append(month)
                errors["RMSE"].append(np.nan)
                errors["MAE"].append(np.nan)
                errors["MBE"].append(np.nan)
                continue

            synthetic = [val * avg_price if not np.isnan(val) else np.nan for val in median_per_hour]
            valid_mask = ~(np.isnan(year_month_data) | np.isnan(synthetic))
            raw_valid = year_month_data[valid_mask]
            synth_valid = np.array(synthetic)[valid_mask]

            if len(raw_valid) > 0 and len(synth_valid) > 0:
                rmse = np.sqrt(mean_squared_error(raw_valid, synth_valid))
                mae = mean_absolute_error(raw_valid, synth_valid)
                mbe = mean_bias_error(raw_valid, synth_valid)
                errors["År"].append(year)
                errors["Måned"].append(month)
                errors["RMSE"].append(round(rmse, 3))
                errors["MAE"].append(round(mae, 3))
                errors["MBE"].append(round(mbe, 3))
            else:
                errors["År"].append(year)
                errors["Måned"].append(month)
                errors["RMSE"].append(np.nan)
                errors["MAE"].append(np.nan)
                errors["MBE"].append(np.nan)

    pd.DataFrame(errors).to_csv(os.path.join(OUTPUT_DIR, "rmse_mae_mbe_per_år_måned.csv"), index=False)
    return errors


def main():
    """Hovedfunksjon for å kjøre analysen."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    df, df_no_2022 = preprocess_data(df)

    # Beregn daglig gjennomsnittspris per måned
    daily_avg = df.groupby(["År", "Måned", "Dag_i_måneden"])["SPOTNO1(Øre/kWh)"].mean().groupby(["År", "Måned"]).mean().to_dict()

    # Normaliser data
    hourly_data = normalize_data(df_no_2022)

    # Generer plott
    median_std_data = plot_boxplot(hourly_data, os.path.join(OUTPUT_DIR, "boxplot_gjennomsnitt_1_uten_2022.png"))
    plot_median_with_std(median_std_data, os.path.join(OUTPUT_DIR, "median_med_std_gjennomsnitt_1_uten_2022.png"))
    plot_yearly_comparison(df, median_std_data, daily_avg)
    plot_selected_months(df, median_std_data, daily_avg)
    calculate_jsd(df, median_std_data, daily_avg)
    calculate_errors(df, median_std_data, daily_avg)

    # Skriv ut bekreftelse
    print(f"Alle plott og tabeller er lagret i: {OUTPUT_DIR}")
    print(f"1. Boxplott: boxplot_gjennomsnitt_1_uten_2022.png")
    print(f"2. Median med std: median_med_std_gjennomsnitt_1_uten_2022.png")
    print("3-12. Sammenligningsplott per år:")
    for year in YEARS:
        print(f"   - År {year}: sammenligning_syntetisk_rådata_med_std_år_{year}.png")
    print("13-15. Sammenligningsplott for januar, juli, oktober:")
    for month in SELECTED_MONTHS:
        print(f"   - {month.capitalize()}: sammenligning_syntetisk_rådata_med_std_{month}_alle_år.png")
    print(f"JSD-plott: jsd_sammenligning_per_år.png")
    print(f"JSD-plott (jan, jul, okt): jsd_sammenligning_jan_jul_okt_per_år.png")
    print(f"Tabeller: median_std_per_time_og_måned.csv, jsd_verdier_per_år.csv, jsd_verdier_jan_jul_okt_per_år.csv, rmse_mae_mbe_per_år_måned.csv")


if __name__ == "__main__":
    main()
