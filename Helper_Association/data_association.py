import pandas as pd
import folium
from folium.plugins import MarkerCluster
from pathlib import Path
import os  
from datetime import timedelta
from pandas import Timedelta
from haversine import haversine, Unit
import gc


# Merge usage count and matched event count for certain keys
def merge_usage_with_matched_crashes(year):    
    usage_file = f"..//results//dataAnalysis//{year}//bike_usage_per_month.csv"
    matched_crash_file = f"..//results//matched_trips//matched_trips_{year}.csv"
    output_file = f"..//results//dataAnalysis//{year}//monthly_usage_with_crashes.csv"
    weekday_output = f"..//results//dataAnalysis//{year}//crash_counts_by_weekday.csv"
    hour_output = f"..//results//dataAnalysis//{year}//crash_counts_by_hour.csv"

    # Step 1: Load matched crash trips
    try:
        crashes = pd.read_csv(matched_crash_file)
        if 'starttime' not in crashes.columns:
            print(f"❌ 'starttime' missing in {matched_crash_file}")
            return

        crashes['starttime'] = pd.to_datetime(crashes['starttime'], errors='coerce')
        crashes = crashes.dropna(subset=['starttime'])
        crashes['year'] = crashes['starttime'].dt.year
        crashes['month'] = crashes['starttime'].dt.month
        crashes['weekday'] = crashes['starttime'].dt.day_name()
        crashes['hour'] = crashes['starttime'].dt.hour
        
        crash_counts = crashes.groupby(['year', 'month']).size().reset_index(name='crash_count')
        
        # Crash counts by weekday
        weekday_counts = crashes['weekday'].value_counts().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ]).fillna(0).astype(int).reset_index()
        weekday_counts.columns = ['weekday', 'crash_count']
        weekday_counts.to_csv(weekday_output, index=False)

        # Crash counts by hour
        hour_counts = crashes['hour'].value_counts().sort_index().reset_index()
        hour_counts.columns = ['hour', 'crash_count']
        hour_counts.to_csv(hour_output, index=False)
        

    except Exception as e:
        print(f"❌ Failed to process crash file for {year}: {e}")
        return

    # Step 2: Load bike usage
    try:
        usage = pd.read_csv(usage_file, header=None if year == '2013' else 'infer')
        
        # Handle missing headers (e.g., for 2013 file)
        if usage.shape[1] == 2:
            usage.columns = ['time', 'trip_count']
        usage['time'] = pd.to_datetime(usage['time'], errors='coerce')
        usage['year'] = usage['time'].dt.year
        usage['month'] = usage['time'].dt.month
        usage = usage.drop(columns=['time'])
        usage['trip_count'] = pd.to_numeric(usage['trip_count'], errors='coerce').fillna(0).astype(int)

    except Exception as e:
        print(f"❌ Failed to load usage data for {year}: {e}")
        return

    # Step 3: Merge
    merged = pd.merge(usage, crash_counts, on=['year', 'month'], how='left')
    merged['crash_count'] = pd.to_numeric(merged['crash_count'], errors='coerce').fillna(0).astype(int)
    merged = merged[merged['trip_count'] > 0]  # Remove any bad rows with 0 trips

    # Step 4: Calculate percentage
    merged['percentage'] = (merged['crash_count'] / merged['trip_count'].replace(0, pd.NA)) * 100
    merged['percentage'] = merged['percentage'].fillna(0)#.round(2)

    # Step 5: Order and Save
    merged = merged[['year', 'month', 'trip_count', 'crash_count', 'percentage']]
    merged = merged.sort_values(by=['year', 'month'])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged.to_csv(output_file, index=False)
    print(f"✅ Saved: {output_file}")


# === Run for all available years ===
def run_for_all_years():
    base_dir = "..//results//dataAnalysis"
    years = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    years.sort()  # Optional: sort years chronologically

    for year in years:
        print(f"\n📅 Processing year: {year}")
        merge_usage_with_matched_crashes(year)



def create_crash_map(df, lat_col, lon_col, datetime_col, id_col, output_file):
    """
    Creates a Folium map with crash markers grouped by year and month.

    Parameters:
        df (pd.DataFrame): Crash data with datetime and coordinates.
        lat_col (str): Column name for latitude.
        lon_col (str): Column name for longitude.
        datetime_col (str): Column name for crash datetime.
        id_col (str): Column name for unique collision ID.
        output_file (str or Path): Path to save the output HTML map.
    """
    # Drop invalid datetimes and coordinates
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    df = df.dropna(subset=[lat_col, lon_col, datetime_col])

    # Extract time parts
    df['year'] = df[datetime_col].dt.year
    df['year_month'] = df[datetime_col].dt.to_period('M').astype(str)

    # Center map on average crash location
    map_center = [df[lat_col].mean(), df[lon_col].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    # === Year-Based Layers ===
    for year, group in df.groupby('year'):
        year_layer = folium.FeatureGroup(name=f"Year: {year}", show=False)
        year_cluster = MarkerCluster().add_to(year_layer)
        
        for _, row in group.iterrows():
            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=f"{id_col}: {row[id_col]}<br>Date: {row[datetime_col].date()}",
            ).add_to(year_cluster)

        year_layer.add_to(m)

    # === Month-Based Layers ===
    for month, group in df.groupby('year_month'):
        month_layer = folium.FeatureGroup(name=f"Month: {month}", show=False)
        month_cluster = MarkerCluster().add_to(month_layer)

        for _, row in group.iterrows():
            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=f"{id_col}: {row[id_col]}<br>Date: {row[datetime_col].date()}",
            ).add_to(month_cluster)

        month_layer.add_to(m)

    # === All Crashes Layer ===
    all_layer = folium.FeatureGroup(name="All Crashes", show=True)
    all_cluster = MarkerCluster().add_to(all_layer)

    for _, row in df.iterrows():
        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=f"{id_col}: {row[id_col]}<br>Date: {row[datetime_col].date()}",
        ).add_to(all_cluster)

    all_layer.add_to(m)

    # Add control and save
    folium.LayerControl(collapsed=False).add_to(m)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_file))
    print(f"✅ Map saved to: {output_file}")
    
    
    

# Match bike involved trips from NYPD to bike trips data from CityBike
def match_crashes_to_trips(
    trip_data_folder,
    crash_file,
    results_folder,
    max_time_diff=1,
    max_distance_m=100,
    start_year=2015,
    end_year=2025
):
    # Load crash data once
    crash_df = pd.read_csv(crash_file)
    crash_df = crash_df[['COLLISION_ID', 'CRASH_DATETIME', 'LATITUDE', 'LONGITUDE']].dropna()
    crash_df['CRASH_DATETIME'] = pd.to_datetime(crash_df['CRASH_DATETIME'], errors='coerce')

    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    for year in range(start_year, end_year + 1):
        trip_file = Path(trip_data_folder) / f"{year}_merged.csv"
        if not trip_file.exists():
            print(f"❌ File not found: {trip_file}")
            continue

        print(f"\n📦 Processing trips for year: {year}")
        trip_df = pd.read_csv(trip_file)
        trip_df['starttime'] = pd.to_datetime(trip_df['starttime'], errors='coerce')
        trip_df['stoptime'] = pd.to_datetime(trip_df['stoptime'], errors='coerce')

        # Filter crash data within time bounds
        earliest_trip_time = trip_df['starttime'].min()
        latest_trip_time = trip_df['starttime'].max()
        time_window_start = earliest_trip_time - Timedelta(minutes=2)
        time_window_end = latest_trip_time + Timedelta(minutes=2)

        crash_df_chunk = crash_df[
            (crash_df['CRASH_DATETIME'] >= time_window_start) &
            (crash_df['CRASH_DATETIME'] <= time_window_end)
        ].copy()

        print(f"🚨 {len(crash_df_chunk)} crash records within trip time range.")

        matched_trip_info = []

        for _, crash_row in crash_df_chunk.iterrows():
            crash_dt = crash_row['CRASH_DATETIME']
            crash_lat = crash_row['LATITUDE']
            crash_lng = crash_row['LONGITUDE']
            crash_id = crash_row['COLLISION_ID']

            window_start = crash_dt - timedelta(minutes=max_time_diff)
            window_end = crash_dt + timedelta(minutes=max_time_diff)

            candidate_trips = trip_df[
                ((trip_df['starttime'] >= window_start) & (trip_df['starttime'] <= window_end)) |
                ((trip_df['stoptime'] >= window_start) & (trip_df['stoptime'] <= window_end))
            ]

            for trip_idx, trip_row in candidate_trips.iterrows():
                match_type = None

                dist_start = haversine(
                    (trip_row['start station latitude'], trip_row['start station longitude']),
                    (crash_lat, crash_lng), unit=Unit.METERS
                ) if pd.notna(trip_row['start station latitude']) and pd.notna(trip_row['start station longitude']) else float('inf')

                dist_end = haversine(
                    (trip_row['end station latitude'], trip_row['end station longitude']),
                    (crash_lat, crash_lng), unit=Unit.METERS
                ) if pd.notna(trip_row['end station latitude']) and pd.notna(trip_row['end station longitude']) else float('inf')

                if dist_start <= max_distance_m and dist_end <= max_distance_m:
                    match_type = 'both'
                elif dist_start <= max_distance_m:
                    match_type = 'start'
                elif dist_end <= max_distance_m:
                    match_type = 'end'

                if match_type:
                    matched_trip_info.append({
                        'trip_idx': trip_idx,
                        'crash_id': crash_id,
                        'match_type': match_type,
                        'dist_start_m': dist_start,
                        'dist_end_m': dist_end,
                        'CRASH_LATITUDE': crash_lat,
                        'CRASH_LONGITUDE': crash_lng,
                        'CRASH_DATETIME': crash_dt
                    })

        # Add crash columns
        trip_df['crash_match_type'] = None
        trip_df['matched_crash_id'] = None
        trip_df['dist_to_crash_start_m'] = None
        trip_df['dist_to_crash_end_m'] = None
        trip_df['CRASH_LATITUDE'] = None
        trip_df['CRASH_LONGITUDE'] = None
        trip_df['CRASH_DATETIME'] = None

        matched_df = pd.DataFrame(matched_trip_info)
        for _, row in matched_df.iterrows():
            trip_df.at[row['trip_idx'], 'crash_match_type'] = row['match_type']
            trip_df.at[row['trip_idx'], 'matched_crash_id'] = row['crash_id']
            trip_df.at[row['trip_idx'], 'dist_to_crash_start_m'] = row['dist_start_m']
            trip_df.at[row['trip_idx'], 'dist_to_crash_end_m'] = row['dist_end_m']
            trip_df.at[row['trip_idx'], 'CRASH_LATITUDE'] = row['CRASH_LATITUDE']
            trip_df.at[row['trip_idx'], 'CRASH_LONGITUDE'] = row['CRASH_LONGITUDE']
            trip_df.at[row['trip_idx'], 'CRASH_DATETIME'] = row['CRASH_DATETIME']

        matched_trips_df = trip_df[trip_df['matched_crash_id'].notna()].copy()
        print(f"✅ Matched trips found: {len(matched_trips_df)}")

        output_file = results_folder / f"matched_trips_{year}.csv"
        matched_trips_df.to_csv(output_file, index=False)
        print(f"💾 Saved to: {output_file}")

        # === Free memory ===
        del trip_df, crash_df_chunk, matched_trip_info, matched_df, matched_trips_df, candidate_trips
        gc.collect()

        
# Merge all matched crash to one file
def merge_all_matched_crash_data(base_dir, output_dir):
   
    crash_files = [f for f in os.listdir(base_dir) if f.startswith("matched_trips_") and f.endswith(".csv")]
    crash_files.sort()
    
    all_crashes = []
    for f in crash_files:
        file_path = os.path.join(base_dir, f)
        print(f"Loading crash data from: {file_path}")
        df = pd.read_csv(file_path)
        df['source_file'] = f  # Optional: keep track of source year/file
        all_crashes.append(df)
    
    if not all_crashes:
        print("❌ No crash data files found.")
        return
    
    combined_crashes = pd.concat(all_crashes, ignore_index=True)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "all_years_matched_crashes.csv")
    combined_crashes.to_csv(output_file, index=False)
    print(f"\n✅ Combined crash data saved: {output_file}")
    print(combined_crashes.head())



# A slightly faster matching method
def match_crashes_to_trips_faster(
    trip_data_folder,
    crash_file,
    results_folder,
    max_time_diff=1,
    max_distance_m=100,
    start_year=2015,
    end_year=2025
):
    # Load crash data once
    crash_df = pd.read_csv(crash_file)
    crash_df = crash_df[['COLLISION_ID', 'CRASH_DATETIME', 'LATITUDE', 'LONGITUDE']].dropna()
    crash_df['CRASH_DATETIME'] = pd.to_datetime(crash_df['CRASH_DATETIME'], errors='coerce')

    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    # Latitude threshold in degrees for pre-filter (approximation)
    lat_threshold_deg = max_distance_m / 111320

    for year in range(start_year, end_year + 1):
        trip_file = Path(trip_data_folder) / f"{year}_merged.csv"
        if not trip_file.exists():
            print(f"❌ File not found: {trip_file}")
            continue

        print(f"\n📦 Processing trips for year: {year}")
        trip_df = pd.read_csv(trip_file)
        trip_df['starttime'] = pd.to_datetime(trip_df['starttime'], errors='coerce')
        trip_df['stoptime'] = pd.to_datetime(trip_df['stoptime'], errors='coerce')

        # Filter crash data within time bounds (±2 minutes buffer)
        earliest_trip_time = trip_df['starttime'].min()
        latest_trip_time = trip_df['starttime'].max()
        time_window_start = earliest_trip_time - Timedelta(minutes=2)
        time_window_end = latest_trip_time + Timedelta(minutes=2)

        crash_df_chunk = crash_df[
            (crash_df['CRASH_DATETIME'] >= time_window_start) &
            (crash_df['CRASH_DATETIME'] <= time_window_end)
        ].copy()

        print(f"🚨 {len(crash_df_chunk)} crash records within trip time range.")

        matched_trip_info = []

        for _, crash_row in crash_df_chunk.iterrows():
            crash_dt = crash_row['CRASH_DATETIME']
            crash_lat = crash_row['LATITUDE']
            crash_lng = crash_row['LONGITUDE']
            crash_id = crash_row['COLLISION_ID']

            window_start = crash_dt - timedelta(minutes=max_time_diff)
            window_end = crash_dt + timedelta(minutes=max_time_diff)

            candidate_trips = trip_df[
                ((trip_df['starttime'] >= window_start) & (trip_df['starttime'] <= window_end)) |
                ((trip_df['stoptime'] >= window_start) & (trip_df['stoptime'] <= window_end))
            ]

            for trip_idx, trip_row in candidate_trips.iterrows():
                match_type = None

                # Pre-check latitude difference before computing haversine distance for start station
                if pd.notna(trip_row['start station latitude']) and pd.notna(trip_row['start station longitude']):
                    if abs(trip_row['start station latitude'] - crash_lat) > lat_threshold_deg:
                        continue  
                    else:
                        dist_start = haversine(
                                    (trip_row['start station latitude'], trip_row['start station longitude']),
                                     (crash_lat, crash_lng), unit=Unit.METERS
                        )
                else:
                    dist_start = float('inf')

                # Pre-check latitude difference before computing haversine distance for end station
                if pd.notna(trip_row['end station latitude']) and pd.notna(trip_row['end station longitude']):
                    if abs(trip_row['end station latitude'] - crash_lat) > lat_threshold_deg:
                        continue  # Skip this trip_row, latitude diff too large
                    else:
                        dist_end = haversine(
                                    (trip_row['end station latitude'], trip_row['end station longitude']),
                                    (crash_lat, crash_lng), unit=Unit.METERS
                            )
                else:
                    dist_end = float('inf')

                if dist_start <= max_distance_m and dist_end <= max_distance_m:
                    match_type = 'both'
                elif dist_start <= max_distance_m:
                    match_type = 'start'
                elif dist_end <= max_distance_m:
                    match_type = 'end'

                if match_type:
                    matched_trip_info.append({
                        'trip_idx': trip_idx,
                        'crash_id': crash_id,
                        'match_type': match_type,
                        'dist_start_m': dist_start,
                        'dist_end_m': dist_end,
                        'CRASH_LATITUDE': crash_lat,
                        'CRASH_LONGITUDE': crash_lng,
                        'CRASH_DATETIME': crash_dt
                    })

        # Initialize crash columns with None
        trip_df['crash_match_type'] = None
        trip_df['matched_crash_id'] = None
        trip_df['dist_to_crash_start_m'] = None
        trip_df['dist_to_crash_end_m'] = None
        trip_df['CRASH_LATITUDE'] = None
        trip_df['CRASH_LONGITUDE'] = None
        trip_df['CRASH_DATETIME'] = None

        matched_df = pd.DataFrame(matched_trip_info)
        for _, row in matched_df.iterrows():
            trip_df.at[row['trip_idx'], 'crash_match_type'] = row['match_type']
            trip_df.at[row['trip_idx'], 'matched_crash_id'] = row['crash_id']
            trip_df.at[row['trip_idx'], 'dist_to_crash_start_m'] = row['dist_start_m']
            trip_df.at[row['trip_idx'], 'dist_to_crash_end_m'] = row['dist_end_m']
            trip_df.at[row['trip_idx'], 'CRASH_LATITUDE'] = row['CRASH_LATITUDE']
            trip_df.at[row['trip_idx'], 'CRASH_LONGITUDE'] = row['CRASH_LONGITUDE']
            trip_df.at[row['trip_idx'], 'CRASH_DATETIME'] = row['CRASH_DATETIME']

        matched_trips_df = trip_df[trip_df['matched_crash_id'].notna()].copy()
        print(f"✅ Matched trips found: {len(matched_trips_df)}")

        output_file = results_folder / f"matched_trips_{year}.csv"
        matched_trips_df.to_csv(output_file, index=False)
        print(f"💾 Saved to: {output_file}")

        # === Free memory ===
        del trip_df, crash_df_chunk, matched_trip_info, matched_df, matched_trips_df, candidate_trips
        gc.collect()
        
        
