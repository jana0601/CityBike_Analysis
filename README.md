# CityBike Analysis

A comprehensive data analysis project exploring New York City's CityBike usage trends and crash associations from **2013 to 2025**. The project involves preprocessing, trip–crash data association, and visualization of rider behavior, station usage, and accident patterns etc.

---

## Workflow Overview

### 1. **Data Preprocessing** (`data_preprocessing/`)
- `S1_Fetch_Data.ipynb` – Downloads raw CityBike and NYPD crash data  
- `S2_UnZip_ALL.ipynb` – Extracts compressed datasets  
- `S3_Process_CrashData_NYPD.ipynb` – Cleans and structures crash data  
- `S4_Final_Merge_Clean.ipynb` – Merges bike and crash datasets for integration  

### 2. **Trip-Crash Association** (`Helper_Association/`)
- `data_association.py` – Core logic to spatially/temporally match trips to cyclist involved crashes  
- `S6_matched_trips.ipynb` – Matches trips to crashes by criteria  
- `S7_Associate_Trip_Crash.ipynb` – Generating monthly usage with crashes & merge all matched trips to one file 
- `S8_plot_maps.ipynb` – Generates interactive crash/usage maps  

### 3. **Analysis & Visualization** (`data_analysis/`)
- `S5_final_bike_data_analysis.ipynb` – Extract some data for analysis from bike dataset such as weekday usage, monthly usage, hourly usage, user type, gender distribution, top 10 stations, long trip summary etc for final statistical and trend analysis  
- `S9_Stats_vis.ipynb` – Visual summaries (gender, usage, time patterns, correlation, prediction etc)  

---

## Key Results

All results are located in the `results/` directory.

###  `results/dataAnalysis/`
Year-by-year statistics from 2013 to 2025, including:
- Age, gender, and user type distributions
- Monthly and hourly usage trends
- Crash occurrences by hour and weekday
- Long-trip station summaries

### `results/extra_analysis/`
Aggregated multi-year plots and insights:
- `bike_usage_per_month_all_years.png`  
- `monthly_crash_rate.png`  
- `gender_distribution_all_years.png`  
- `trip_duration_distribution.png`  
- `matched_crashes_map_by_month_and_year.html` (interactive map)

### `results/matched_trips/`
- `matched_trips_YYYY.csv`: Trip-level data linked to crashes  
- `all_years_matched_crashes.csv`: Consolidated crash associations  

---

## Env Used


- Jupyter Notebooks with Python 3
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `folium`

---

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/jana0601/CityBike_Analysis.git
cd CityBike_Analysis
Install required packages (using pip or conda):
pip install -r requirements.txt 
Run preprocessing notebooks in order:

S1 → S4 in data_preprocessing/

S6 → S8 in Helper_Association/

