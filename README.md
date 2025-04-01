<a href="https://www.linkedin.com/in/kshitija-chilbule-b98515309/" target="_blank">
  <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin" alt="LinkedIn Badge" style="height: 30px; width: auto;">
</a>

# 📈 Forecasting Tesla Stock Prices Using Time Series Analysis 🚗
Welcome to the <b>Tesla Stock Price Forecasting project</b>, where we delve into time-series analysis to predict stock price trends for one of the world's most innovative companies—Tesla Inc.

# 🛠️ Tech Stack
## `Python🐍` `Time-Series🎢`

## 📜Table of Contents 
- [Objective](#objective)
- [Project Flow](#project-flow)
- [Models Used](#models-used)
- [Conclusion](#conclusion)

## 🎯Objective
This project aims to analyze the Tesla stock price dataset by performing detailed <b>Exploratory Data Analysis (EDA)</b> and developing robust forecasting models to predict future stock trends. This can provide valuable insights for investors and stakeholders.

## 🔢Models Used
We leverage the following time-series forecasting models:

- <b>ARIMA:</b>  Autoregressive Integrated Moving Average
- <b>SARIMA:</b> Seasonal Autoregressive Integrated Moving Average

## 🌟Project Flow
### 1. Data Ingestion
  
![image](https://github.com/user-attachments/assets/b8823c44-b4ee-43c0-975a-e6aaeb062e64)

In this project, I am performing an univariate analysis on "Date" and "Close" features

![image](https://github.com/user-attachments/assets/ce3856de-725c-404c-b824-6abe1001f422)

### 2. Exploratory Data Analysis
Visualizing Trends for closing price

![image](https://github.com/user-attachments/assets/8d633544-3af2-49b6-9d63-248f928d48d1)

We performed visualization-based testing and statistics-based testing to check whether the data is stationary or non-stationary.

#### 1. Results of visualization-based testing
![image](https://github.com/user-attachments/assets/67eb2888-85e5-457a-8ecc-b74bc99f3090)
- From the image, we can see that,
   - The data is following an upward trend
   - The mean and the standard deviation of the data are not constant.

#### 2. Results of Augmented Dickey-Fuller (ADF) Test
![image](https://github.com/user-attachments/assets/addb58b6-c728-43ce-805f-9d53d9f6da8b)
![image](https://github.com/user-attachments/assets/f71c0b9b-f1e0-484c-83ff-db9117678237)
- As we can see, the P value is less than 0.05

<b>Result: </b> The overall result of both tests suggests that this time series data is non-stationary, and hence we accept the null hypothesis

#### Decomposition of Time Series

##### 1. Additive Decomposition Series
- When the variations in the components remain constant over time.

![image](https://github.com/user-attachments/assets/9d28e9d6-8adc-4b25-bb25-8ab5a5158327)

Where,
- T(t): Trend (long-term movement)
- S(t): Seasonality (repeating patterns)
- N(t): Noise / Residuals (random fluctuations)

![image](https://github.com/user-attachments/assets/505d2e31-c229-4771-8b64-b696881f1cb2)

##### 2. Multiplicative Decomposition Series
- When variations increase or decrease proportionally over time.

![image](https://github.com/user-attachments/assets/8c2fb618-e801-4572-a434-2d2b2864b2e1)

Where,
- T(t): Trend (long-term movement)
- S(t): Seasonality (repeating patterns)
- N(t): Noise / Residuals (random fluctuations)

![image](https://github.com/user-attachments/assets/aaae5e72-1715-4545-a8c0-0620cf2e3db9)

#### Relationship between observations in a Time Series data
<b>1. ACF (Autocorrelation Function): </b> 
- ACF measures the correlation between a time series and its past values (lags).
- It tells how strongly the current value is related to past values.

![image](https://github.com/user-attachments/assets/9e701e95-fe2d-49d0-a846-287a78d3a9d6)
- The autocorrelation values do not drop off quickly; instead, they remain strong across many lags.
- A slow decline suggests a strong trend in the data.
- ACF does not cut off at a specific lag → No clear Moving Average (q) order.

<b>2. PACF (Partial Autocorrelation Function): </b>
-  PACF measures the direct correlation between a time series and its past values, removing the effect of intermediate lags.
-  PACF (Partial Autocorrelation Function) plot helps in determining the AR (Auto-Regressive) order for an ARIMA model
![image](https://github.com/user-attachments/assets/69218881-4c37-49ee-b8b5-1108485c6131)

- No significant spikes after Lag 2 (values stay within the shaded region), meaning past values beyond Lag 2 don’t directly impact the present.
- Since PACF has a sharp drop after Lag 2, it indicates that an AR(2) model might be a good choice.

### 3. Data Transformation
The process of converting non-stationary data into stationary data

<b> Using Differencing Method: </b> 
- Differencing removes trend & makes data stationary
- Prepares the data for ARIMA modeling
- Improves model performance for time series forecasting
  
In this problem statement, we applied first-order differencing, meaning each value is replaced with the difference between itself and the previous value.

![image](https://github.com/user-attachments/assets/8c9411b4-880e-4bcc-af7e-bcf44df65405)

![image](https://github.com/user-attachments/assets/3d9de77e-9886-47fa-8634-6534afad5751)

We ran an ADF test (Augmented Dickey-Fuller Test) after differencing. Below is a result
![image](https://github.com/user-attachments/assets/c35a109c-4747-40c5-90ac-64bdf22573a1)

- P value is now greater than 0.05, hence we will reject the null hypothesis
- The mean and Standard deviation of the data are now constant
- Now our time series data is stationary

<b> After conversion of non-stationary data into stationary data, we again plotted a ACF graph </b>
![image](https://github.com/user-attachments/assets/d6bb227d-d294-418c-8584-168bd4025d14)

- Since the ACF cuts off sharply at lag 1, this suggests:
  - The data has a Moving Average component of order 1 (q=1)

### 4. Model Building
#### 1. ARIMA (AutoRegressive Integrated Moving Average)
Now that we have got values of p,d, and q, we can apply the  ARIMA model to our data
- p = 2
- d = 1
- q = 1

<b>Note:</b> In actual code, we have implied permutation combination to choose the best p,d,q values.

![image](https://github.com/user-attachments/assets/298d26e3-3619-4b61-84ca-2dfa6c19e5af)

![image](https://github.com/user-attachments/assets/94a5d2d7-cf58-432c-8248-5982d72f4f4b)


#### 2. SARIMA (Seasonal AutoRegressive Integrated Moving Average)
![image](https://github.com/user-attachments/assets/839dfabd-3a9e-43c0-9242-84652b0534e5)

## 📌Conclusion
The project demonstrates how time-series models like ARIMA and SARIMA can effectively forecast stock price movements, providing essential insights for decision-making.

# Connect 🤝
- 📩 <b>Email:</b> kshitijachilbule5@gmail.com
- 📶 <b>LinkedIn:</b> https://www.linkedin.com/in/kshitija-chilbule-b98515309/
