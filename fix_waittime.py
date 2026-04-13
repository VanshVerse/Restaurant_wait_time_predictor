import pandas as pd
import numpy as np

np.random.seed(42)
n = 1200

day_of_week      = np.random.randint(0, 7, n)
hour_of_day      = np.random.randint(10, 23, n)
party_size       = np.random.randint(1, 11, n)
tables_occupied  = np.random.randint(5, 51, n)
tables_total     = 50
staff_on_duty    = np.random.randint(3, 16, n)
is_weekend       = (day_of_week >= 5).astype(int)
is_holiday       = np.random.choice([0,1], n, p=[0.85, 0.15])
reservations_ahead = np.random.randint(0, 30, n)
avg_service_time   = np.random.uniform(30, 90, n)
weather_score    = np.random.randint(1, 6, n)
occupancy_pct    = tables_occupied / tables_total

# New realistic formula — max 25 min
wait_time = (
    0.8  * party_size +
    0.2  * tables_occupied +
    3.0  * is_weekend +
    2.0  * is_holiday +
    0.3  * reservations_ahead +
    0.05 * avg_service_time +
    0.5  * weather_score -
    0.4  * staff_on_duty +
    np.where((hour_of_day >= 12) & (hour_of_day <= 14), 2, 0) +
    np.where((hour_of_day >= 19) & (hour_of_day <= 21), 3, 0) +
    np.random.normal(0, 1.5, n)
).clip(0, 25)

df = pd.DataFrame({
    'day_of_week'         : day_of_week,
    'hour_of_day'         : hour_of_day,
    'party_size'          : party_size,
    'tables_occupied'     : tables_occupied,
    'staff_on_duty'       : staff_on_duty,
    'is_weekend'          : is_weekend,
    'is_holiday'          : is_holiday,
    'reservations_ahead'  : reservations_ahead,
    'avg_service_time_min': np.round(avg_service_time, 1),
    'weather_score'       : weather_score,
    'occupancy_pct'       : np.round(occupancy_pct, 3),
    'wait_time_min'       : np.round(wait_time, 1),
})

df.to_csv('restaurant_wait.csv', index=False)
print(f"✅ New CSV created!")
print(f"Min: {df['wait_time_min'].min()} | Max: {df['wait_time_min'].max()} | Mean: {df['wait_time_min'].mean():.1f}")