# API Examples - /predict Endpoint

**Note:** Only basic features are required. Derived features (cyclical encodings, interactions, etc.) are computed automatically.

**Important:** The prediction values shown below are **real examples** from the model. Your predictions will vary based on the input features you provide.

## What do the predictions mean?

The API returns **predicted hourly bike rental demand** (number of bikes). For example:
- `"prediction": 16.94` means the model predicts **~17 bikes** will be rented in that hour
- `"prediction": 20.44` means the model predicts **~20 bikes** will be rented in that hour

These are **counts** (número de bicicletas) that will be rented in the specified hour based on the weather, time, and other features provided.

## Required Basic Features

- `temp`: Temperature (normalized, 0-1)
- `atemp`: Feels-like temperature (normalized, 0-1)
- `hum`: Humidity (normalized, 0-1)
- `windspeed`: Wind speed (normalized, 0-1)
- `season`: Season (1=spring, 2=summer, 3=fall, 4=winter)
- `yr`: Year (0=2011, 1=2012)
- `mnth`: Month (1-12)
- `weathersit`: Weather situation (1=clear, 2=mist, 3=light rain/snow, 4=heavy rain/snow)
- `hr`: Hour of day (0-23)
- `weekday`: Day of week (0=Sunday, 6=Saturday)
- `holiday`: Is holiday (0=no, 1=yes)
- `workingday`: Is working day (0=no, 1=yes)

---

## Single Prediction

```json
{
  "features": {
    "temp": 0.3,
    "atemp": 0.32,
    "hum": 0.5,
    "windspeed": 0.2,
    "season": 2,
    "yr": 1,
    "mnth": 6,
    "weathersit": 1,
    "hr": 14,
    "weekday": 1,
    "holiday": 0,
    "workingday": 1
  }
}
```

**Response:**
```json
{
  "prediction": 20.113128662109375
}
```

*Note: This is a real prediction from the model. Values will vary based on input features.*

---

## Batch Prediction

```json
{
  "records": [
    {
      "temp": 0.3,
      "atemp": 0.32,
      "hum": 0.5,
      "windspeed": 0.2,
      "season": 2,
      "yr": 1,
      "mnth": 6,
      "weathersit": 1,
      "hr": 14,
      "weekday": 1,
      "holiday": 0,
      "workingday": 1
    },
    {
      "temp": 0.5,
      "atemp": 0.52,
      "hum": 0.6,
      "windspeed": 0.3,
      "season": 3,
      "yr": 1,
      "mnth": 9,
      "weathersit": 2,
      "hr": 18,
      "weekday": 2,
      "holiday": 0,
      "workingday": 1
    }
  ]
}
```

**Response:**
```json
{
    "predictions": [
        20.113128662109375,
        20.40223503112793
    ]
}
```

*Note: These are real predictions from the model. Actual values will vary based on input features.*

---

**Note:** Features that require historical data (lags, rolling stats) are automatically set to default values. Check `/health` endpoint for model information.

