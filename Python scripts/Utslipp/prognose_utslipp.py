
"""
Genererer prognoser for utslippsintensitet (g CO₂/kWh) i Norge for 2025–2030 basert på
historiske data fra 2015–2024. Skriptet behandler timebaserte utslippsdata, lager syntetiske
profiler og genererer visualiseringer for årlige, månedlige og daglige prognoser.

"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Konfigurasjon
DATA_PATH = Path("/Users/pathujahvimalasenan/Desktop/Master/Data/CO2/Alle land/Alle_timer_2015-2024_utslippsintensitet.csv")
OUTPUT_DIR = Path("/Users/pathujahvimalasenan/Desktop/Master/Data/CO2/grafer/Progoser/")
FONT_SIZE = 20
MONTH_NAMES = ["jan", "feb", "mar", "apr", "mai", "jun", "jul", "aug", "sep", "okt", "nov", "des"]
MONTH_NAMES_SHORT = [m[:3] for m in MONTH_NAMES]
HOUR_NAMES = [str(t) for t in range(24)]
DAY_NAMES = [str(d) for d in range(1, 32)]
HISTORICAL_YEARS = list(range(2015, 2025))
FORECAST_YEARS = list(range(2025, 2031))
COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
Y_AXIS_RANGES = {
    'yearly_avg_emissions': (0, 400),
    'monthly_forecast': (0, 300),
    'monthly_day_forecast': (0, 410),
    'daily_hour_forecast': (0, 310),
}

def load_and_preprocess_data(file_path: Path) -> pd.DataFrame:
    """Laster og forbehandler CSV-datafilen."""
    if not file_path.exists():
        raise FileNotFoundError(f"Fant ikke datafilen på {file_path}")
    
    df = pd.read_csv(file_path)
    def parse_datetime(row):
        return pd.Timestamp(int(row['År']), int(row['Måned']), int(row['Dag']), int(row['time']))
    
    df['Time'] = df.apply(parse_datetime, axis=1)
    df = df[df['Time'].dt.year != 2025]
    
    df['Year'] = df['Time'].dt.year
    df['Month'] = df['Time'].dt.month
    df['Day'] = df['Time'].dt.day
    df['Hour'] = df['Time'].dt.hour
    return df

def create_synthetic_profiles(df: pd.DataFrame) -> tuple[np.ndarray, dict, dict, dict]:
    """Genererer syntetiske profiler for årlige, månedlige og daglige utslippsvariasjoner."""
    scaler = MinMaxScaler()
    
    # Årlig profil
    monthly_avg = df.groupby(['Year', 'Month'])['Utslippsintensitet'].mean().reset_index()
    monthly_avg['Normalized_Emissions'] = np.nan
    for year in HISTORICAL_YEARS:
        mask = monthly_avg['Year'] == year
        data = monthly_avg.loc[mask, 'Utslippsintensitet'].values.reshape(-1, 1)
        if len(data) > 0:
            monthly_avg.loc[mask, 'Normalized_Emissions'] = scaler.fit_transform(data).flatten()
    
    monthly_avg['Normalized_Relative'] = np.nan
    for month in range(1, 13):
        mask = monthly_avg['Month'] == month
        emissions = monthly_avg.loc[mask, 'Normalized_Emissions']
        mean_emissions = np.nanmean(emissions)
        if mean_emissions != 0:
            monthly_avg.loc[mask, 'Normalized_Relative'] = emissions / mean_emissions
    
    yearly_median = np.array([np.median(monthly_avg[monthly_avg['Month'] == m]['Normalized_Relative'].dropna()) 
                             for m in range(1, 13)])
    mean_median = np.mean(yearly_median)
    if mean_median != 0:
        yearly_median = yearly_median / mean_median
    
    # Månedlig profil
    daily_avg = df.groupby(['Year', 'Month', 'Day'])['Utslippsintensitet'].mean().reset_index()
    daily_avg['Normalized_Emissions'] = np.nan
    for year in HISTORICAL_YEARS:
        for month in range(1, 13):
            mask = (daily_avg['Year'] == year) & (daily_avg['Month'] == month)
            data = daily_avg.loc[mask, 'Utslippsintensitet'].values.reshape(-1, 1)
            if len(data) > 0:
                daily_avg.loc[mask, 'Normalized_Emissions'] = scaler.fit_transform(data).flatten()
    
    mean_emissions = np.nanmean(daily_avg['Normalized_Emissions'])
    daily_avg['Normalized_Emissions_Scaled'] = daily_avg['Normalized_Emissions'] / mean_emissions if mean_emissions != 0 else np.nan
    
    monthly_median = {}
    monthly_std = {}
    for month in range(1, 13):
        month_data = daily_avg[daily_avg['Month'] == month]
        day_data = [month_data[month_data['Day'] == d]['Normalized_Emissions_Scaled'].values 
                    for d in range(1, 32)]
        monthly_median[MONTH_NAMES[month-1]] = [np.nanmedian(data) if len(data) > 0 else np.nan for data in day_data]
        monthly_std[MONTH_NAMES[month-1]] = [np.nanstd(data) if len(data) > 0 else np.nan for data in day_data]
    
    # Daglig profil
    hourly_avg = df.groupby(['Year', 'Month', 'Hour'])['Utslippsintensitet'].mean().reset_index()
    hourly_avg['Normalized_Emissions'] = np.nan
    for year in HISTORICAL_YEARS:
        for month in range(1, 13):
            mask = (hourly_avg['Year'] == year) & (hourly_avg['Month'] == month)
            data = hourly_avg.loc[mask, 'Utslippsintensitet'].values
            if len(data) > 0 and (mean_data := np.nanmean(data)) != 0:
                hourly_avg.loc[mask, 'Normalized_Emissions'] = data / mean_data
    
    daily_median = {}
    daily_std = {}
    for month in range(1, 13):
        month_data = hourly_avg[hourly_avg['Month'] == month]
        hour_data = [month_data[month_data['Hour'] == h]['Normalized_Emissions'].dropna().values 
                     for h in range(24)]
        daily_median[MONTH_NAMES[month-1]] = [np.median(data) if len(data) > 0 else np.nan for data in hour_data]
        daily_std[MONTH_NAMES[month-1]] = [np.std(data) if len(data) > 0 else np.nan for data in hour_data]
    
    return yearly_median, monthly_median, daily_median, daily_std

def generate_forecasts(df: pd.DataFrame, yearly_median: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genererer månedlige og årlige utslippsprognoser for 2025–2030."""
    global_mean_emissions = 211.5  # g CO₂/kWh for 2025
    monthly_forecast = []
    for month in range(1, 13):
        for year in FORECAST_YEARS:
            trend_factor = 1 - 0.0342 * (year - 2025)  # 3.42% reduksjon per år
            adjusted_mean = global_mean_emissions * trend_factor
            synthetic_factor = yearly_median[month-1] if not np.isnan(yearly_median[month-1]) else 1.0
            forecast_emissions = max(adjusted_mean * synthetic_factor, 24.0)  # Minimum 24 g CO₂/kWh
            monthly_forecast.append([year, month, forecast_emissions])
    
    monthly_forecast_df = pd.DataFrame(monthly_forecast, columns=['Year', 'Month', 'Utslippsintensitet'])
    yearly_forecast_df = monthly_forecast_df.groupby('Year')['Utslippsintensitet'].mean().reset_index()
    yearly_forecast_df.columns = ['Year', 'Gjennomsnittlig Årsutslipp (g CO₂/kWh)']
    
    return monthly_forecast_df, yearly_forecast_df

def plot_yearly_emissions(historical_df: pd.DataFrame, forecast_df: pd.DataFrame, output_dir: Path):
    """Lager plott for gjennomsnittlig årlig utslippsintensitet (2015–2030)."""
    combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    historical_data = combined_df[combined_df['Year'] <= 2024]
    
    plt.figure(figsize=(10, 6))
    plt.plot(historical_data['Year'].to_numpy(), historical_data['Gjennomsnittlig Årsutslipp (g CO₂/kWh)'].to_numpy(), 
             color='black', linewidth=2, marker='o', label='Historisk')
    plt.plot(combined_df['Year'].to_numpy(), combined_df['Gjennomsnittlig Årsutslipp (g CO₂/kWh)'].to_numpy(), 
             color='blue', linewidth=2, marker='o', label='Prognose', zorder=1)
    
    plt.title('Gjennomsnittlig Årsutslipp (2015-2030)', fontsize=FONT_SIZE)
    plt.xlabel('År', fontsize=FONT_SIZE)
    plt.ylabel('Utslippsintensitet (g CO₂/kWh)', fontsize=FONT_SIZE)
    plt.xticks(range(2015, 2031), rotation=45, fontsize=FONT_SIZE-4)
    plt.yticks(fontsize=FONT_SIZE-4)
    plt.grid(True)
    plt.legend(fontsize=FONT_SIZE-4)
    if 'yearly_avg_emissions' in Y_AXIS_RANGES:
        plt.ylim(Y_AXIS_RANGES['yearly_avg_emissions'])
    plt.tight_layout()
    plt.savefig(output_dir / 'gjennomsnittlig_årsutslipp_prognose.png')
    plt.close()

def plot_monthly_forecast(forecast_df: pd.DataFrame, output_dir: Path):
    """Lager plott for månedlige utslippsprognoser (2025–2030)."""
    plt.figure(figsize=(10, 6))
    for idx, year in enumerate(FORECAST_YEARS):
        emissions = [
            forecast_df[(forecast_df['Year'] == year) & (forecast_df['Month'] == m)]['Utslippsintensitet'].iloc[0]
            if not forecast_df[(forecast_df['Year'] == year) & (forecast_df['Month'] == m)].empty else np.nan 
            for m in range(1, 13)
        ]
        plt.plot(range(1, 13), emissions, label=f'År {year}', color=COLORS[idx], linewidth=2)
    
    plt.title('Årsprognoser 2025-2030', fontsize=FONT_SIZE)
    plt.xlabel('Måned', fontsize=FONT_SIZE)
    plt.ylabel('Utslippsintensitet (g CO₂/kWh)', fontsize=FONT_SIZE)
    plt.xticks(range(1, 13), MONTH_NAMES_SHORT, rotation=45, ha='right', fontsize=FONT_SIZE-4)
    plt.yticks(fontsize=FONT_SIZE-4)
    plt.grid(True)
    plt.legend(title="År", fontsize=FONT_SIZE-4, title_fontsize=FONT_SIZE-4)
    if 'monthly_forecast' in Y_AXIS_RANGES:
        plt.ylim(Y_AXIS_RANGES['monthly_forecast'])
    plt.tight_layout()
    plt.savefig(output_dir / 'prognose_alle_år_utslipp.png')
    plt.close()

def plot_monthly_day_forecast(forecast_df: pd.DataFrame, monthly_median: dict, output_dir: Path):
    """Lager plott for månedlige dagbaserte prognoser (2025–2030)."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 20), sharex=True, sharey=True)
    axes = axes.flatten()
    
    if 'monthly_day_forecast' not in Y_AXIS_RANGES:
        all_y_values = []
        for month in range(1, 13):
            profile = monthly_median[MONTH_NAMES[month-1]]
            for year in FORECAST_YEARS:
                mean_emissions = forecast_df[
                    (forecast_df['Year'] == year) & (forecast_df['Month'] == month)
                ]['Utslippsintensitet'].iloc[0] if not forecast_df[
                    (forecast_df['Year'] == year) & (forecast_df['Month'] == month)
                ].empty else 0
                scaled = [val * mean_emissions for val in profile]
                all_y_values.extend([val for val in scaled if not np.isnan(val)])
        y_min, y_max = min(all_y_values), max(all_y_values)
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
    else:
        y_min, y_max = Y_AXIS_RANGES['monthly_day_forecast']
    
    for month in range(1, 13):
        profile = monthly_median[MONTH_NAMES[month-1]]
        valid_positions = [i for i, val in enumerate(profile) if not np.isnan(val)]
        for idx, year in enumerate(FORECAST_YEARS):
            mean_emissions = forecast_df[
                (forecast_df['Year'] == year) & (forecast_df['Month'] == month)
            ]['Utslippsintensitet'].iloc[0] if not forecast_df[
                (forecast_df['Year'] == year) & (forecast_df['Month'] == month)
            ].empty else 0
            scaled = [val * mean_emissions for val in profile]
            axes[month-1].plot(valid_positions, [scaled[i] for i in valid_positions], 
                               color=COLORS[idx], label=f'År {year}', linewidth=1)
        axes[month-1].set_title(MONTH_NAMES[month-1], fontsize=FONT_SIZE-2)
        axes[month-1].grid(True)
        axes[month-1].tick_params(axis='y', labelsize=FONT_SIZE-4)
        axes[month-1].set_ylim(y_min, y_max)
        if month == 1:
            axes[month-1].legend(fontsize=FONT_SIZE-6)
    
    for ax in axes:
        ax.set_xticks(range(31))
        ax.set_xticklabels(DAY_NAMES, rotation=45, ha='right', fontsize=FONT_SIZE-10)
    
    fig.suptitle('Månedsprognoser 2025-2030', fontsize=FONT_SIZE+3)
    fig.text(0.5, 0.04, 'Dag i måneden', ha='center', fontsize=FONT_SIZE+3)
    fig.text(0.04, 0.5, 'Utslippsintensitet (g CO₂/kWh)', va='center', rotation='vertical', fontsize=FONT_SIZE+3)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
    plt.savefig(output_dir / 'prognose_måned_gjennomsnitt_dag_utslipp.png')
    plt.close()

def plot_daily_hour_forecast(forecast_df: pd.DataFrame, daily_median: dict, output_dir: Path):
    """Lager plott for daglige timebaserte prognoser (2025–2030)."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 20), sharex=True, sharey=True)
    axes = axes.flatten()
    
    if 'daily_hour_forecast' not in Y_AXIS_RANGES:
        all_y_values = []
        for month in range(1, 13):
            profile = daily_median[MONTH_NAMES[month-1]]
            for year in FORECAST_YEARS:
                mean_emissions = forecast_df[
                    (forecast_df['Year'] == year) & (forecast_df['Month'] == month)
                ]['Utslippsintensitet'].iloc[0] if not forecast_df[
                    (forecast_df['Year'] == year) & (forecast_df['Month'] == month)
                ].empty else 0
                scaled = [val * mean_emissions for val in profile]
                all_y_values.extend([val for val in scaled if not np.isnan(val)])
        y_min, y_max = min(all_y_values), max(all_y_values)
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
    else:
        y_min, y_max = Y_AXIS_RANGES['daily_hour_forecast']
    
    for month in range(1, 13):
        profile = daily_median[MONTH_NAMES[month-1]]
        for idx, year in enumerate(FORECAST_YEARS):
            mean_emissions = forecast_df[
                (forecast_df['Year'] == year) & (forecast_df['Month'] == month)
            ]['Utslippsintensitet'].iloc[0] if not forecast_df[
                (forecast_df['Year'] == year) & (forecast_df['Month'] == month)
            ].empty else 0
            scaled = [val * mean_emissions for val in profile]
            axes[month-1].plot(range(24), scaled, color=COLORS[idx], label=f'År {year}', linewidth=1)
        axes[month-1].set_title(MONTH_NAMES[month-1], fontsize=FONT_SIZE-2)
        axes[month-1].grid(True)
        axes[month-1].tick_params(axis='y', labelsize=FONT_SIZE-4)
        axes[month-1].set_ylim(y_min, y_max)
        if month == 1:
            axes[month-1].legend(fontsize=FONT_SIZE-4)
    
    for ax in axes:
        ax.set_xticks(range(24))
        ax.set_xticklabels(HOUR_NAMES, rotation=45, ha='right', fontsize=FONT_SIZE-10)
    
    fig.suptitle('Døgnprognoser 2025-2030', fontsize=FONT_SIZE+3)
    fig.text(0.5, 0.04, 'Time i døgnet', ha='center', fontsize=FONT_SIZE+3)
    fig.text(0.04, 0.5, 'Utslippsintensitet (g CO₂/kWh)', va='center', rotation='vertical', fontsize=FONT_SIZE+3)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
    plt.savefig(output_dir / 'prognose_døgn_gjennomsnitt_time_utslipp.png')
    plt.close()

def main():
    """Hovedfunksjon for å kjøre hele prognosegenereringsprosessen."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Last og forbehandle data
    df = load_and_preprocess_data(DATA_PATH)
    
    # Generer syntetiske profiler
    yearly_median, monthly_median, daily_median, daily_std = create_synthetic_profiles(df)
    
    # Generer prognoser
    monthly_forecast_df, yearly_forecast_df = generate_forecasts(df, yearly_median)
    
    # Generer historiske årsutslipp
    monthly_avg = df.groupby(['Year', 'Month'])['Utslippsintensitet'].mean().reset_index()
    historical_yearly_df = monthly_avg.groupby('Year')['Utslippsintensitet'].mean().reset_index()
    historical_yearly_df.columns = ['Year', 'Gjennomsnittlig Årsutslipp (g CO₂/kWh)']
    
    # Lag plott
    plot_yearly_emissions(historical_yearly_df, yearly_forecast_df, OUTPUT_DIR)
    plot_monthly_forecast(monthly_forecast_df, OUTPUT_DIR)
    plot_monthly_day_forecast(monthly_forecast_df, monthly_median, OUTPUT_DIR)
    plot_daily_hour_forecast(monthly_forecast_df, daily_median, OUTPUT_DIR)
    
    # Lagre prognosetabell
    forecast_table = pd.DataFrame({
        'Year': FORECAST_YEARS,
        'Prognose': yearly_forecast_df['Gjennomsnittlig Årsutslipp (g CO₂/kWh)'].round(2)
    })
    forecast_table.to_csv(OUTPUT_DIR / 'prognose_gjennomsnitt_utslipp.csv', index=False)
    
    # Skriv ut filstier
    print("Alle plott og tabeller er lagret i:", OUTPUT_DIR)
    print("\nGenererte filer:")
    print(f"  - Årlig utslippsprognose: {OUTPUT_DIR / 'gjennomsnittlig_årsutslipp_prognose.png'}")
    print(f"  - Månedlig prognose: {OUTPUT_DIR / 'prognose_alle_år_utslipp.png'}")
    print(f"  - Månedlig dagprognose: {OUTPUT_DIR / 'prognose_måned_gjennomsnitt_dag_utslipp.png'}")
    print(f"  - Daglig timeprognose: {OUTPUT_DIR / 'prognose_døgn_gjennomsnitt_time_utslipp.png'}")
    print(f"  - Prognosetabell: {OUTPUT_DIR / 'prognose_gjennomsnitt_utslipp.csv'}")

if __name__ == "__main__":
    main()