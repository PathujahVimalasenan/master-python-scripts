#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genererer prognoser for strømpriser i Norge (2025-2030) basert på historiske data fra NVE.
Skriptet behandler timebaserte prisdata, lager syntetiske profiler og genererer visualiseringer
for årlige, månedlige og daglige prisprognoser.

Forfatter: Pathujah Vimalasenan
Opprettet: 5. mai 2025
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Konfigurasjon
DATASTI = Path("/Users/pathujahvimalasenan/Desktop/Master/Data/Historiske priser NVE/timesdata_pris_2015-2025.xlsx")
LAGRINGSSTI = Path("/Users/pathujahvimalasenan/Desktop/Master/Data/grafer/prognoser1/")
TEKSTSTØRRELSE = 20
MÅNEDSNAVN = ["jan", "feb", "mar", "apr", "mai", "jun", "jul", "aug", "sep", "okt", "nov", "des"]
MÅNEDSNAVN_KORT = [m[:3] for m in MÅNEDSNAVN]
TIMENAVN = [str(t) for t in range(24)]
DAGNAVN = [str(d) for d in range(1, 32)]
HISTORISKE_ÅR = list(range(2015, 2025))
PROGNOSE_ÅR = list(range(2025, 2031))
FARGER = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
Y_AKSEGRENSER = {
    'årlig_gjennomsnittspris': (0, 200),
    'månedlig_prognose': (0, 150),
    'månedlig_dag_prognose': (0, 150),
    'daglig_time_prognose': (0, 150),
}

def last_og_forbehandle_data(filsti: Path) -> pd.DataFrame:
    """Laster og forbehandler Excel-datafilen."""
    if not filsti.exists():
        raise FileNotFoundError(f"Fant ikke datafilen på {filsti}")
    
    df = pd.read_excel(filsti)
    df['Tid'] = pd.to_datetime(df['Hour'], format='%d.%m.%Y %H:%M')
    df = df[df['Tid'].dt.year != 2025]
    
    df['År'] = df['Tid'].dt.year
    df['Måned'] = df['Tid'].dt.month
    df['Dag'] = df['Tid'].dt.day
    df['Time'] = df['Tid'].dt.hour
    return df

def lag_syntetiske_profiler(df: pd.DataFrame) -> tuple[list, dict, dict, dict]:
    """Genererer syntetiske profiler for årlige, månedlige og daglige prisvariasjoner."""
    df_uten_2022 = df[df['År'] != 2022].copy()
    
    # Årlig profil
    månedlig_gjennomsnitt = df_uten_2022.groupby(['År', 'Måned'])['SPOTNO1(Øre/kWh)'].mean().reset_index()
    månedlig_gjennomsnitt['Normalisert_Pris'] = np.nan
    for år in HISTORISKE_ÅR:
        if år == 2022:
            continue
        mask = månedlig_gjennomsnitt['År'] == år
        priser = månedlig_gjennomsnitt.loc[mask, 'SPOTNO1(Øre/kWh)'].values
        if len(priser) > 0 and (gjennomsnitt := np.mean(priser)) != 0:
            månedlig_gjennomsnitt.loc[mask, 'Normalisert_Pris'] = priser / gjennomsnitt
    
    årlig_median = [np.median(månedlig_gjennomsnitt[månedlig_gjennomsnitt['Måned'] == m]['Normalisert_Pris'].dropna()) 
                    for m in range(1, 13)]
    årlig_std = [np.std(månedlig_gjennomsnitt[månedlig_gjennomsnitt['Måned'] == m]['Normalisert_Pris'].dropna(), ddof=1) 
                 for m in range(1, 13)]
    
    # Månedlig profil
    daglig_gjennomsnitt = df_uten_2022.groupby(['År', 'Måned', 'Dag'])['SPOTNO1(Øre/kWh)'].mean().reset_index()
    daglig_gjennomsnitt['Normalisert_Pris'] = np.nan
    for år in HISTORISKE_ÅR:
        if år == 2022:
            continue
        for måned in range(1, 13):
            mask = (daglig_gjennomsnitt['År'] == år) & (daglig_gjennomsnitt['Måned'] == måned)
            priser = daglig_gjennomsnitt.loc[mask, 'SPOTNO1(Øre/kWh)'].values
            if len(priser) > 0 and (gjennomsnitt := np.mean(priser)) != 0:
                daglig_gjennomsnitt.loc[mask, 'Normalisert_Pris'] = priser / gjennomsnitt
    
    månedlig_median = {}
    månedlig_std = {}
    for måned in range(1, 13):
        måned_data = daglig_gjennomsnitt[daglig_gjennomsnitt['Måned'] == måned]
        dag_data = [måned_data[måned_data['Dag'] == d]['Normalisert_Pris'].dropna().values 
                    for d in range(1, 32)]
        månedlig_median[MÅNEDSNAVN[måned-1]] = [np.median(data) if len(data) > 0 else np.nan for data in dag_data]
        månedlig_std[MÅNEDSNAVN[måned-1]] = [np.std(data, ddof=1) if len(data) > 0 else np.nan for data in dag_data]
    
    # Daglig profil
    timelig_gjennomsnitt = df_uten_2022.groupby(['År', 'Måned', 'Time'])['SPOTNO1(Øre/kWh)'].mean().reset_index()
    timelig_gjennomsnitt['Normalisert_Pris'] = np.nan
    for år in HISTORISKE_ÅR:
        if år == 2022:
            continue
        for måned in range(1, 13):
            mask = (timelig_gjennomsnitt['År'] == år) & (timelig_gjennomsnitt['Måned'] == måned)
            priser = timelig_gjennomsnitt.loc[mask, 'SPOTNO1(Øre/kWh)'].values
            if len(priser) > 0 and (gjennomsnitt := np.nanmean(priser)) != 0:
                timelig_gjennomsnitt.loc[mask, 'Normalisert_Pris'] = priser / gjennomsnitt
    
    daglig_median = {}
    daglig_std = {}
    for måned in range(1, 13):
        måned_data = timelig_gjennomsnitt[timelig_gjennomsnitt['Måned'] == måned]
        time_data = [måned_data[måned_data['Time'] == t]['Normalisert_Pris'].dropna().values 
                     for t in range(24)]
        daglig_median[MÅNEDSNAVN[måned-1]] = [np.median(data) if len(data) > 0 else np.nan for data in time_data]
        daglig_std[MÅNEDSNAVN[måned-1]] = [np.std(data, ddof=1) if len(data) > 0 else np.nan for data in time_data]
    
    return årlig_median, månedlig_median, daglig_median, daglig_std

def generer_prognoser(df: pd.DataFrame, årlig_median: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genererer månedlige og årlige prisprognoser for 2025-2030."""
    årlig_gjennomsnitt = df.groupby(['År'])['SPOTNO1(Øre/kWh)'].mean().reset_index()
    pris_2024 = årlig_gjennomsnitt[årlig_gjennomsnitt['År'] == 2024]['SPOTNO1(Øre/kWh)'].iloc[0]
    pris_2030 = 80.0  # øre/kWh
    
    år_intervall = np.array(range(2024, 2031))
    pris_trend = np.linspace(pris_2024, pris_2030, len(år_intervall))
    trend_dict = dict(zip(år_intervall, pris_trend))
    
    prognose_månedlig = []
    for måned in range(1, 13):
        for år in PROGNOSE_ÅR:
            global_pris = trend_dict[år]
            syntetisk_faktor = årlig_median[måned-1] if not np.isnan(årlig_median[måned-1]) else 1.0
            prognose_pris = max(global_pris * syntetisk_faktor, 1.0)
            prognose_månedlig.append([år, måned, prognose_pris])
    
    prognose_månedlig_df = pd.DataFrame(prognose_månedlig, columns=['År', 'Måned', 'SPOTNO1(Øre/kWh)'])
    prognose_årlig_df = prognose_månedlig_df.groupby('År')['SPOTNO1(Øre/kWh)'].mean().reset_index()
    prognose_årlig_df.columns = ['År', 'Gjennomsnittlig Årspris (Øre/kWh)']
    
    return prognose_månedlig_df, prognose_årlig_df

def lag_årlig_prisplott(historisk_df: pd.DataFrame, prognose_df: pd.DataFrame, lagringssti: Path):
    """Lager plott for gjennomsnittlig årlig pris (2015-2030)."""
    kombinert_df = pd.concat([historisk_df, prognose_df], ignore_index=True)
    
    plt.figure(figsize=(10, 6))
    # Konverter Pandas Series til NumPy-arrays for å unngå indekseringsfeil
    plt.plot(kombinert_df['År'].to_numpy(), kombinert_df['Gjennomsnittlig Årspris (Øre/kWh)'].to_numpy(), 
             color='black', linewidth=2, marker='o', label='Historisk (2015-2024)')
    plt.plot(prognose_df['År'].to_numpy(), prognose_df['Gjennomsnittlig Årspris (Øre/kWh)'].to_numpy(), 
             color='blue', linewidth=2, marker='o', label='Prognose (2025-2030)')
    
    plt.title('Gjennomsnittlig Kraftpris i Norge (2015-2030)', fontsize=TEKSTSTØRRELSE)
    plt.xlabel('År', fontsize=TEKSTSTØRRELSE)
    plt.ylabel('Pris (Øre/kWh)', fontsize=TEKSTSTØRRELSE)
    plt.xticks(range(2015, 2031), rotation=45, fontsize=TEKSTSTØRRELSE-4)
    plt.yticks(fontsize=TEKSTSTØRRELSE-4)
    plt.grid(True)
    plt.legend(fontsize=TEKSTSTØRRELSE-4)
    if 'årlig_gjennomsnittspris' in Y_AKSEGRENSER:
        plt.ylim(Y_AKSEGRENSER['årlig_gjennomsnittspris'])
    plt.tight_layout()
    plt.savefig(lagringssti / 'gjennomsnittlig_årspris.png')
    plt.close()

def lag_månedlig_prognoseplott(prognose_df: pd.DataFrame, lagringssti: Path):
    """Lager plott for månedlige prisprognoser (2025-2030)."""
    plt.figure(figsize=(10, 6))
    for idx, år in enumerate(PROGNOSE_ÅR):
        priser = [
            prognose_df[(prognose_df['År'] == år) & (prognose_df['Måned'] == m)]['SPOTNO1(Øre/kWh)'].iloc[0]
            if not prognose_df[(prognose_df['År'] == år) & (prognose_df['Måned'] == m)].empty else np.nan 
            for m in range(1, 13)
        ]
        plt.plot(range(1, 13), priser, label=f'År {år}', color=FARGER[idx], linewidth=2)
    
    plt.title('Prognose for Månedspriser (2025-2030)', fontsize=TEKSTSTØRRELSE)
    plt.xlabel('Måned', fontsize=TEKSTSTØRRELSE)
    plt.ylabel('Pris (Øre/kWh)', fontsize=TEKSTSTØRRELSE)
    plt.xticks(range(1, 13), MÅNEDSNAVN_KORT, rotation=45, ha='right', fontsize=TEKSTSTØRRELSE-4)
    plt.yticks(fontsize=TEKSTSTØRRELSE-4)
    plt.grid(True)
    plt.legend(title="År", fontsize=TEKSTSTØRRELSE-4, title_fontsize=TEKSTSTØRRELSE-4)
    if 'månedlig_prognose' in Y_AKSEGRENSER:
        plt.ylim(Y_AKSEGRENSER['månedlig_prognose'])
    plt.tight_layout()
    plt.savefig(lagringssti / 'prognose_alle_år.png')
    plt.close()

def lag_månedlig_dag_prognoseplott(prognose_df: pd.DataFrame, månedlig_median: dict, lagringssti: Path):
    """Lager plott for månedlige dagbaserte prognoser (2025-2030)."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 20), sharex=True, sharey=True)
    axes = axes.flatten()
    
    if 'månedlig_dag_prognose' not in Y_AKSEGRENSER:
        alle_y_verdier = []
        for måned in range(1, 13):
            profil = månedlig_median[MÅNEDSNAVN[måned-1]]
            for år in PROGNOSE_ÅR:
                gjennomsnitt = prognose_df[
                    (prognose_df['År'] == år) & (prognose_df['Måned'] == måned)
                ]['SPOTNO1(Øre/kWh)'].iloc[0] if not prognose_df[
                    (prognose_df['År'] == år) & (prognose_df['Måned'] == måned)
                ].empty else 0
                skalert = [val * gjennomsnitt for val in profil]
                alle_y_verdier.extend([val for val in skalert if not np.isnan(val)])
        y_min, y_max = min(alle_y_verdier), max(alle_y_verdier)
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
    else:
        y_min, y_max = Y_AKSEGRENSER['månedlig_dag_prognose']
    
    for måned in range(1, 13):
        profil = månedlig_median[MÅNEDSNAVN[måned-1]]
        gyldige_posisjoner = [i for i, val in enumerate(profil) if not np.isnan(val)]
        for idx, år in enumerate(PROGNOSE_ÅR):
            gjennomsnitt = prognose_df[
                (prognose_df['År'] == år) & (prognose_df['Måned'] == måned)
            ]['SPOTNO1(Øre/kWh)'].iloc[0] if not prognose_df[
                (prognose_df['År'] == år) & (prognose_df['Måned'] == måned)
            ].empty else 0
            skalert = [val * gjennomsnitt for val in profil]
            axes[måned-1].plot(gyldige_posisjoner, [skalert[i] for i in gyldige_posisjoner], 
                               color=FARGER[idx], label=f'År {år}', linewidth=1)
        axes[måned-1].set_title(MÅNEDSNAVN[måned-1], fontsize=TEKSTSTØRRELSE-2)
        axes[måned-1].grid(True)
        axes[måned-1].tick_params(axis='y', labelsize=TEKSTSTØRRELSE-4)
        axes[måned-1].set_ylim(y_min, y_max)
        if måned == 1:
            axes[måned-1].legend(fontsize=TEKSTSTØRRELSE-4)
    
    for ax in axes:
        ax.set_xticks(range(31))
        ax.set_xticklabels(DAGNAVN, rotation=45, ha='right', fontsize=TEKSTSTØRRELSE-10)
    
    fig.suptitle('Månedsprognoser 2025-2030', fontsize=TEKSTSTØRRELSE+3)
    fig.text(0.5, 0.04, 'Dag i måneden', ha='center', fontsize=TEKSTSTØRRELSE+3)
    fig.text(0.04, 0.5, 'Pris (Øre/kWh)', va='center', rotation='vertical', fontsize=TEKSTSTØRRELSE+3)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
    plt.savefig(lagringssti / 'prognose_måned_gjennomsnitt_dag.png')
    plt.close()

def lag_daglig_time_prognoseplott(prognose_df: pd.DataFrame, daglig_median: dict, lagringssti: Path):
    """Lager plott for daglige timebaserte prognoser (2025-2030)."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 20), sharex=True, sharey=True)
    axes = axes.flatten()
    
    if 'daglig_time_prognose' not in Y_AKSEGRENSER:
        alle_y_verdier = []
        for måned in range(1, 13):
            profil = daglig_median[MÅNEDSNAVN[måned-1]]
            for år in PROGNOSE_ÅR:
                gjennomsnitt = prognose_df[
                    (prognose_df['År'] == år) & (prognose_df['Måned'] == måned)
                ]['SPOTNO1(Øre/kWh)'].iloc[0] if not prognose_df[
                    (prognose_df['År'] == år) & (prognose_df['Måned'] == måned)
                ].empty else 0
                skalert = [val * gjennomsnitt for val in profil]
                alle_y_verdier.extend([val for val in skalert if not np.isnan(val)])
        y_min, y_max = min(alle_y_verdier), max(alle_y_verdier)
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
    else:
        y_min, y_max = Y_AKSEGRENSER['daglig_time_prognose']
    
    for måned in range(1, 13):
        profil = daglig_median[MÅNEDSNAVN[måned-1]]
        for idx, år in enumerate(PROGNOSE_ÅR):
            gjennomsnitt = prognose_df[
                (prognose_df['År'] == år) & (prognose_df['Måned'] == måned)
            ]['SPOTNO1(Øre/kWh)'].iloc[0] if not prognose_df[
                (prognose_df['År'] == år) & (prognose_df['Måned'] == måned)
            ].empty else 0
            skalert = [val * gjennomsnitt for val in profil]
            axes[måned-1].plot(range(24), skalert, color=FARGER[idx], label=f'År {år}', linewidth=1)
        axes[måned-1].set_title(MÅNEDSNAVN[måned-1], fontsize=TEKSTSTØRRELSE-2)
        axes[måned-1].grid(True)
        axes[måned-1].tick_params(axis='y', labelsize=TEKSTSTØRRELSE-4)
        axes[måned-1].set_ylim(y_min, y_max)
        if måned == 1:
            axes[måned-1].legend(fontsize=TEKSTSTØRRELSE-4)
    
    for ax in axes:
        ax.set_xticks(range(24))
        ax.set_xticklabels(TIMENAVN, rotation=45, ha='right', fontsize=TEKSTSTØRRELSE-10)
    
    fig.suptitle('Døgnprognoser 2025-2030', fontsize=TEKSTSTØRRELSE+3)
    fig.text(0.5, 0.04, 'Time i døgnet', ha='center', fontsize=TEKSTSTØRRELSE+3)
    fig.text(0.04, 0.5, 'Pris (Øre/kWh)', va='center', rotation='vertical', fontsize=TEKSTSTØRRELSE+3)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
    plt.savefig(lagringssti / 'prognose_døgn_gjennomsnitt_time.png')
    plt.close()

def main():
    """Hovedfunksjon for å kjøre hele prognosegenereringsprosessen."""
    os.makedirs(LAGRINGSSTI, exist_ok=True)
    
    # Last og forbehandle data
    df = last_og_forbehandle_data(DATASTI)
    
    # Generer syntetiske profiler
    årlig_median, månedlig_median, daglig_median, daglig_std = lag_syntetiske_profiler(df)
    
    # Generer prognoser
    prognose_månedlig_df, prognose_årlig_df = generer_prognoser(df, årlig_median)
    
    # Generer historiske årspriser
    månedlig_gjennomsnitt = df.groupby(['År', 'Måned'])['SPOTNO1(Øre/kWh)'].mean().reset_index()
    historisk_årlig_df = månedlig_gjennomsnitt.groupby('År')['SPOTNO1(Øre/kWh)'].mean().reset_index()
    historisk_årlig_df.columns = ['År', 'Gjennomsnittlig Årspris (Øre/kWh)']
    
    # Lag plott
    lag_årlig_prisplott(historisk_årlig_df, prognose_årlig_df, LAGRINGSSTI)
    lag_månedlig_prognoseplott(prognose_månedlig_df, LAGRINGSSTI)
    lag_månedlig_dag_prognoseplott(prognose_månedlig_df, månedlig_median, LAGRINGSSTI)
    lag_daglig_time_prognoseplott(prognose_månedlig_df, daglig_median, LAGRINGSSTI)
    
    # Lagre prognosetabell
    prognose_tabell = pd.DataFrame({
        'År': PROGNOSE_ÅR,
        'Gjennomsnittlig Årspris (Øre/kWh)': prognose_årlig_df['Gjennomsnittlig Årspris (Øre/kWh)'].round(2)
    })
    prognose_tabell.to_csv(LAGRINGSSTI / 'prognose_gjennomsnitt.csv', index=False)
    
    # Skriv ut filstier
    print("Alle plott og tabeller er lagret i:", LAGRINGSSTI)
    print("\nGenererte filer:")
    print(f"  - Gjennomsnittlig årspris: {LAGRINGSSTI / 'gjennomsnittlig_årspris.png'}")
    print(f"  - Månedsprognoser: {LAGRINGSSTI / 'prognose_alle_år.png'}")
    print(f"  - Månedlig dagprognose: {LAGRINGSSTI / 'prognose_måned_gjennomsnitt_dag.png'}")
    print(f"  - Daglig timeprognose: {LAGRINGSSTI / 'prognose_døgn_gjennomsnitt_time.png'}")
    print(f"  - Prognosetabell: {LAGRINGSSTI / 'prognose_gjennomsnitt.csv'}")

if __name__ == "__main__":
    main()