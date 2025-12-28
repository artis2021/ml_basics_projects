import requests, pandas as pd, matplotlib.pyplot as plt, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

API_KEY = "my_api_key"
CITY = "Delhi"

url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

response = requests.get(url)
data = response.json()

print(data)

weather_data = {
    "temperature": data["main"]["temp"],
    "humidity": data["main"]["humidity"],
    "pressure": data["main"]["pressure"],
    "wind_speed": data["wind"]["speed"]
}

print(weather_data)

df = pd.DataFrame([weather_data])
df.to_csv("weather_data.csv", mode="a", header=False, index=False)

data = pd.read_csv(
    "weather_data.csv",
    names=["temperature", "humidity", "pressure", "wind_speed"]
)

print(data.head())

data = data.dropna()

X = data[["humidity", "pressure", "wind_speed"]]
y = data["temperature"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

error = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", error)

plt.scatter(y_test, predictions)
plt.xlabel("Actual Temp")
plt.ylabel("Predicted Temp")
plt.title("Weather Forecast Prediction")
plt.show()

new_weather = pd.DataFrame([{
    "humidity": weather_data["humidity"],
    "pressure": weather_data["pressure"],
    "wind_speed": weather_data["wind_speed"],
}])

predicted_temp = model.predict(new_weather)
print("Presicted Temp:", predicted_temp[0])