import requests, numpy as np, pandas as pd, matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

API_KEY = "my_api_key"
CITY = "Delhi"

url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
response = requests.get(url)
data = response.json()

print(data)

rain_data = {
    "temperature": data["main"]["temp"],
    "humidity": data["main"]["humidity"],
    "pressure": data["main"]["pressure"],
    "wind_speed": data["wind"]["speed"],
    "rainfall": data.get("rain", {}).get("1h", 0)
}

print("Current weather data: ", rain_data)

df = pd.DataFrame([rain_data])
df.to_csv("rainfall_data.csv", mode="a", header=False, index=False)

dataset = pd.read_csv(
    "rainfall_data.csv",
    names=["temperature", "humidity", "pressure", "wind_speed", "rainfall"]
)

dataset = dataset.dropna()
print(dataset)

X = dataset[["temperature", "humidity", "pressure", "wind_speed"]]
y = dataset["rainfall"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error (Rainfall mm): ", error)

plt.scatter(y_test, predictions)
plt.xlabel("Actual Rainfall (mm)")
plt.ylabel("Predicted Rainfall (mm)")
plt.title("Rainfall Prediction Model")
plt.show()

current_weather = pd.DataFrame([{
    "temperature": rain_data["temperature"],
    "humidity": rain_data["humidity"],
    "pressure": rain_data["pressure"],
    "wind_speed": rain_data["wind_speed"],
}])

predicted_rainfall = model.predict(current_weather)

print("Predicted Rainfall (mm):", max(0, predicted_rainfall[0]))
print(predicted_rainfall)