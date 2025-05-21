import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

def forecast_arima(series, horizon):
    """
    Forecasts future values using an ARIMA model.
    Returns original series and forecast.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception as e:
            print(f"Error converting series index to DatetimeIndex for ARIMA: {e}")
            return None, None

    original_freq = series.index.freq
    if original_freq is None:
        original_freq = pd.infer_freq(series.index)
        if original_freq is None:
            print("Warning: Could not infer frequency for ARIMA, defaulting to 'D'. This may affect results.")
            original_freq = 'D' # Default to daily if cannot infer
        series = series.asfreq(original_freq)
        # Fill NaNs that might have been introduced by asfreq
        series.fillna(method='ffill', inplace=True)
        series.fillna(method='bfill', inplace=True)


    if len(series) < 10: # Minimum length for ARIMA
        print("Error: Series is too short for ARIMA modeling.")
        return series, None # Return original series and None for forecast part
    
    try:
        # Using a common order (p,d,q). This may need tuning or auto_arima.
        order = (5, 1, 0) 
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast_values = model_fit.forecast(steps=horizon)
        
        last_date = series.index[-1]
        forecast_index = pd.date_range(start=last_date, periods=horizon + 1, freq=original_freq)[1:]
        forecast_series = pd.Series(forecast_values, index=forecast_index)
        
        return series, forecast_series
    except Exception as e:
        print(f"Error during ARIMA forecasting: {e}")
        return series, None

def forecast_exponential_smoothing(series, horizon):
    """
    Forecasts future values using an Exponential Smoothing model (Holt-Winters).
    Returns original series and forecast.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception as e:
            print(f"Error converting series index to DatetimeIndex for ExpSmoothing: {e}")
            return None, None

    original_freq = series.index.freq
    if original_freq is None:
        original_freq = pd.infer_freq(series.index)
        if original_freq is None:
            print("Warning: Could not infer frequency for ExpSmoothing, defaulting to 'D'. This may affect results.")
            original_freq = 'D' # Default to daily
        series = series.asfreq(original_freq)
        series.fillna(method='ffill', inplace=True)
        series.fillna(method='bfill', inplace=True)
            
    seasonal_periods = None
    freq_str = str(original_freq).upper() # Ensure freq_str is a string and uppercase

    if 'D' in freq_str or 'B' in freq_str: # Daily or business daily
        if len(series) >= 14: seasonal_periods = 7 
    elif 'W' in freq_str: # Weekly
         if len(series) >= 8: seasonal_periods = 4 
    elif 'M' in freq_str: # Monthly
        if len(series) >= 24: seasonal_periods = 12
            
    # Check length against seasonal_periods; minimum 2 * seasonal_periods if seasonal
    min_len = 2 * seasonal_periods if seasonal_periods else 2
    if len(series) < min_len:
        print(f"Warning: Series is too short for seasonal Exponential Smoothing (len: {len(series)}, seasonal_periods: {seasonal_periods}). Forcing non-seasonal or simple model.")
        seasonal_periods = None # Fallback to non-seasonal
        if len(series) < 2:
             print("Error: Series is too short for any Exponential Smoothing model.")
             return series, None


    try:
        if seasonal_periods:
            model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=seasonal_periods, initialization_method='estimated')
        else:
            model = ExponentialSmoothing(series, trend='add', initialization_method='estimated') 
            
        model_fit = model.fit()
        forecast_values = model_fit.forecast(steps=horizon)
        
        last_date = series.index[-1]
        forecast_index = pd.date_range(start=last_date, periods=horizon + 1, freq=original_freq)[1:]
        forecast_series = pd.Series(forecast_values, index=forecast_index)
        
        return series, forecast_series
    except Exception as e:
        print(f"Error during Exponential Smoothing forecasting: {e}")
        return series, None

AVAILABLE_MODELS = {
    "ARIMA": forecast_arima,
    "Exponential Smoothing": forecast_exponential_smoothing,
}
