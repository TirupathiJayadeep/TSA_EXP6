# Ex.No: 6 HOLT WINTERS METHOD
### Date: 28-08-2026
### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```python
# IMPORTING NECESSARY LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------
# LOAD THE DATASET
# ---------------------------------------------------

data = pd.read_csv("Month_Value_1.csv")

print(data.head())
print(data.info())


# ---------------------------------------------------
# CONVERT PERIOD COLUMN INTO DATETIME
# ---------------------------------------------------

data['Period'] = pd.to_datetime(
    data['Period'],
    format='%d.%m.%Y'
)

# Set Period as index
data.set_index('Period', inplace=True)

# Sort according to date
data.sort_index(inplace=True)

print(data.head())


# ---------------------------------------------------
# SELECT TARGET COLUMN
# ---------------------------------------------------

data_monthly = data['Revenue'].dropna()

print("\nMonthly Revenue Data:")
print(data_monthly.head())

print("\nNumber of available observations:")
print(len(data_monthly))


# ---------------------------------------------------
# PLOT THE ORIGINAL TIME SERIES
# ---------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    data_monthly,
    label='Revenue'
)

plt.title('Monthly Revenue')
plt.xlabel('Period')
plt.ylabel('Revenue')
plt.legend()
plt.grid()

plt.show()


# ---------------------------------------------------
# SEASONAL DECOMPOSITION
# ---------------------------------------------------

decomposition = seasonal_decompose(
    data_monthly,
    model='additive',
    period=12
)

decomposition.plot()
plt.suptitle("Seasonal Decomposition of Revenue", y=1.02)
plt.show()


# ---------------------------------------------------
# SPLIT TRAIN AND TEST DATA
# ---------------------------------------------------

train_size = int(len(data_monthly) * 0.8)

train_data = data_monthly.iloc[:train_size]
test_data = data_monthly.iloc[train_size:]

print("Training data size:", len(train_data))
print("Testing data size:", len(test_data))


# ---------------------------------------------------
# CREATE HOLT-WINTERS MODEL
# ---------------------------------------------------

model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()


# ---------------------------------------------------
# PREDICT TEST DATA
# ---------------------------------------------------

test_predictions = model.forecast(
    steps=len(test_data)
)


# ---------------------------------------------------
# VISUAL EVALUATION
# ---------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    train_data,
    label='Training Data'
)

plt.plot(
    test_data,
    label='Actual Test Data'
)

plt.plot(
    test_predictions,
    label='Predicted Test Data'
)

plt.title('Holt-Winters Model - Test Prediction')
plt.xlabel('Period')
plt.ylabel('Revenue')
plt.legend()
plt.grid()

plt.show()


# ---------------------------------------------------
# MODEL PERFORMANCE METRICS
# ---------------------------------------------------

mae = mean_absolute_error(
    test_data,
    test_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        test_data,
        test_predictions
    )
)

print("MAE:", mae)
print("RMSE:", rmse)


# ---------------------------------------------------
# CREATE FINAL MODEL USING ALL AVAILABLE DATA
# ---------------------------------------------------

final_model = ExponentialSmoothing(
    data_monthly,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()


# ---------------------------------------------------
# PREDICT FUTURE DATA
# ---------------------------------------------------

future_steps = 32

final_predictions = final_model.forecast(
    steps=future_steps
)


# ---------------------------------------------------
# PLOT FINAL PREDICTION
# ---------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    data_monthly,
    label='Actual Revenue'
)

plt.plot(
    final_predictions,
    label='Future Prediction'
)

plt.title('Holt-Winters Revenue Forecast')
plt.xlabel('Period')
plt.ylabel('Revenue')
plt.legend()
plt.grid()

plt.show()


# ---------------------------------------------------
# DISPLAY FUTURE PREDICTIONS
# ---------------------------------------------------

future_forecast = pd.DataFrame({
    'Predicted_Revenue': final_predictions
})

print("\nFuture Revenue Predictions:")
print(future_forecast)
```
### OUTPUT:
<img width="842" height="718" alt="image" src="https://github.com/user-attachments/assets/be64f9ea-a83c-481b-8c56-4776b5a9a2dc" />

<img width="1012" height="672" alt="image" src="https://github.com/user-attachments/assets/f0e861d2-3571-41fe-987a-b034927e5315" />

<img width="712" height="557" alt="image" src="https://github.com/user-attachments/assets/8d257e4b-dc3c-48d8-8aa1-6d230b2c51ef" />

<img width="990" height="517" alt="image" src="https://github.com/user-attachments/assets/c3ab2de4-2f2a-4361-b580-4df494754df3" />

<img width="1002" height="481" alt="image" src="https://github.com/user-attachments/assets/310c5aa3-5f12-4f84-92b7-6a5d42b6d4ef" />

<img width="297" height="592" alt="image" src="https://github.com/user-attachments/assets/8679dae1-2921-4882-9cf7-da450c66ea2f" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
