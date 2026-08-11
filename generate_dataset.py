import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_traffic_dataset():
    np.random.seed(42)
    start_date = datetime(2023, 1, 1, 0, 0, 0)
    num_hours = 24 * 365 # 1 full year of hourly records = 8,760 hours
    
    dates = [start_date + timedelta(hours=i) for i in range(num_hours)]
    
    holidays_list = [
        ("2023-01-01", "New Year's Day"),
        ("2023-01-16", "Martin Luther King Jr. Day"),
        ("2023-02-20", "Washington's Birthday"),
        ("2023-05-29", "Memorial Day"),
        ("2023-07-04", "Independence Day"),
        ("2023-09-04", "Labor Day"),
        ("2023-10-09", "Columbus Day"),
        ("2023-11-11", "Veterans Day"),
        ("2023-11-23", "Thanksgiving"),
        ("2023-12-25", "Christmas Day")
    ]
    holiday_dict = {h[0]: h[1] for h in holidays_list}
    
    weather_mains = ["Clear", "Clouds", "Rain", "Snow", "Mist", "Fog", "Drizzle", "Thunderstorm"]
    weather_descriptions = {
        "Clear": ["sky is clear"],
        "Clouds": ["few clouds", "scattered clouds", "broken clouds", "overcast clouds"],
        "Rain": ["light rain", "moderate rain", "heavy intensity rain"],
        "Snow": ["light snow", "heavy snow"],
        "Mist": ["mist"],
        "Fog": ["fog"],
        "Drizzle": ["light intensity drizzle"],
        "Thunderstorm": ["thunderstorm with light rain", "proximity thunderstorm"]
    }
    
    data = []
    
    for dt in dates:
        date_str = dt.strftime("%Y-%m-%d")
        hour = dt.hour
        month = dt.month
        day_of_week = dt.weekday() # 0 = Mon, 6 = Sun
        is_weekend = 1 if day_of_week >= 5 else 0
        
        holiday = holiday_dict.get(date_str, "No Holiday")
        
        # Temperature in Celsius (seasonal curve + diurnal variation + noise)
        base_temp = 10 + 15 * np.sin((month - 4) * np.pi / 6) # -5°C in winter to 25°C in summer
        diurnal_temp = 4 * np.sin((hour - 9) * np.pi / 12)
        temp_celsius = round(base_temp + diurnal_temp + np.random.normal(0, 2), 1)
        
        # Weather selection
        weather_prob = [0.40, 0.30, 0.12, 0.05 if month in [11, 12, 1, 2, 3] else 0.0, 0.05, 0.03, 0.03, 0.02]
        prob_sum = sum(weather_prob)
        norm_prob = [p / prob_sum for p in weather_prob]
        
        weather_main = np.random.choice(weather_mains, p=norm_prob)
        weather_desc = np.random.choice(weather_descriptions[weather_main])
        
        # Rain and Snow
        rain_1h = 0.0
        if weather_main == "Rain":
            rain_1h = round(float(np.random.exponential(scale=2.5)), 2)
        elif weather_main == "Drizzle":
            rain_1h = round(float(np.random.uniform(0.1, 1.0)), 2)
            
        snow_1h = 0.0
        if weather_main == "Snow":
            snow_1h = round(float(np.random.exponential(scale=1.5)), 2)
            
        # Cloud cover
        if weather_main == "Clear":
            clouds_all = np.random.randint(0, 20)
        elif weather_main == "Clouds":
            clouds_all = np.random.randint(30, 100)
        else:
            clouds_all = np.random.randint(70, 100)
            
        # Traffic Volume calculation based on realistic factors
        # Base hourly volume profile (rush hours 7-9 AM, 4-6 PM)
        if hour in [7, 8, 9]:
            base_volume = 4800 + np.random.normal(0, 300)
        elif hour in [16, 17, 18]:
            base_volume = 5400 + np.random.normal(0, 350)
        elif hour in [10, 11, 12, 13, 14, 15]:
            base_volume = 3500 + np.random.normal(0, 250)
        elif hour in [19, 20, 21]:
            base_volume = 2800 + np.random.normal(0, 200)
        else: # Late night / early morning 22-6
            base_volume = 700 + np.random.normal(0, 150)
            
        # Modifiers
        if is_weekend:
            base_volume *= 0.65 # Weekend traffic reduction
            if hour in [12, 13, 14, 15, 16, 17]: # Weekend afternoon bump
                base_volume *= 1.2
                
        if holiday != "No Holiday":
            base_volume *= 0.45 # Major holiday reduction
            
        # Weather impact
        if weather_main in ["Rain", "Drizzle"]:
            base_volume *= 0.88 # Bad weather reduces volume or slows traffic
        elif weather_main == "Snow":
            base_volume *= 0.70
        elif weather_main == "Thunderstorm":
            base_volume *= 0.60
        elif weather_main in ["Fog", "Mist"]:
            base_volume *= 0.92
            
        # Temperature extreme impact
        if temp_celsius < -5 or temp_celsius > 35:
            base_volume *= 0.95
            
        traffic_volume = max(100, int(round(base_volume)))
        
        data.append({
            "date": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": hour,
            "holiday": holiday,
            "temperature": temp_celsius,
            "rain_1h": rain_1h,
            "snow_1h": snow_1h,
            "clouds_all": clouds_all,
            "weather_main": weather_main,
            "weather_description": weather_desc,
            "traffic_volume": traffic_volume
        })
        
    df = pd.DataFrame(data)
    df.to_csv("data/traffic_data.csv", index=False)
    print(f"Dataset generated successfully with {len(df)} records at data/traffic_data.csv")

if __name__ == "__main__":
    generate_traffic_dataset()
