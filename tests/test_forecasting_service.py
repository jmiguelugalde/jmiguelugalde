import pytest
import pandas as pd
from pandas.testing import assert_series_equal
from datetime import datetime
import numpy as np

# Adjust import path if necessary
from services.forecasting_service import forecast_arima, forecast_exponential_smoothing, AVAILABLE_MODELS

@pytest.fixture
def sample_daily_series():
    """Creates a sample daily time series for 30 days."""
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    data = np.random.randint(50, 200, size=30)
    series = pd.Series(data, index=dates, dtype=float)
    return series

@pytest.fixture
def short_daily_series():
    """Creates a very short time series (5 days)."""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    data = np.random.randint(50, 200, size=5)
    series = pd.Series(data, index=dates, dtype=float)
    return series

@pytest.fixture
def sample_monthly_series():
    """Creates a sample monthly time series for 36 months."""
    dates = pd.date_range(start='2020-01-01', periods=36, freq='MS') # MS for Month Start
    data = np.random.randint(1000, 5000, size=36) + np.arange(36) * 50 # Add some trend
    # Add some seasonality
    for i in range(len(data)):
        if dates[i].month in [11, 12]: # Higher sales in Nov, Dec
            data[i] *= 1.5
        if dates[i].month in [1, 2]: # Lower sales in Jan, Feb
            data[i] *= 0.8
    series = pd.Series(data, index=dates, dtype=float)
    return series


HORIZON = 12

# --- Tests for forecast_arima ---
def test_arima_returns_series_and_forecast(sample_daily_series):
    """Test that forecast_arima returns original series and a forecast series."""
    original_series, forecast = forecast_arima(sample_daily_series.copy(), HORIZON)
    
    assert isinstance(original_series, pd.Series), "Original series should be a pandas Series"
    assert_series_equal(original_series, sample_daily_series) # Check if original is returned as is or correctly processed

    assert isinstance(forecast, pd.Series), "Forecast should be a pandas Series"
    assert len(forecast) == HORIZON, f"Forecast length should be equal to horizon {HORIZON}"
    assert forecast.dtype == float, "Forecast series should contain float values"
    assert isinstance(forecast.index, pd.DatetimeIndex), "Forecast series index should be DatetimeIndex"
    # Check if forecast index follows original series index
    expected_forecast_start_date = sample_daily_series.index[-1] + pd.Timedelta(days=1) # Assuming daily frequency
    assert forecast.index[0] == expected_forecast_start_date, "Forecast index does not start correctly."


def test_arima_short_series(short_daily_series):
    """Test ARIMA with a very short series (should return original series and None for forecast)."""
    original_series, forecast = forecast_arima(short_daily_series.copy(), HORIZON)
    assert_series_equal(original_series, short_daily_series)
    assert forecast is None, "Forecast should be None for very short series where model can't fit"

def test_arima_series_with_no_freq(sample_daily_series):
    """Test ARIMA when series index has no frequency set initially."""
    series_no_freq = sample_daily_series.copy()
    series_no_freq.index.freq = None # Explicitly remove frequency
    
    original_series, forecast = forecast_arima(series_no_freq, HORIZON)
    
    assert isinstance(forecast, pd.Series), "Forecast should still be generated for series without explicit freq"
    assert len(forecast) == HORIZON
    # The service should infer daily frequency 'D' or handle it.
    # Check if the forecast index has a frequency (it should after processing by the service)
    assert forecast.index.freq is not None, "Forecast index should have a frequency after processing"


# --- Tests for forecast_exponential_smoothing ---
def test_exp_smoothing_returns_series_and_forecast(sample_daily_series):
    """Test that forecast_exponential_smoothing returns original series and a forecast series."""
    original_series, forecast = forecast_exponential_smoothing(sample_daily_series.copy(), HORIZON)

    assert isinstance(original_series, pd.Series), "Original series should be a pandas Series"
    assert_series_equal(original_series, sample_daily_series)

    assert isinstance(forecast, pd.Series), "Forecast should be a pandas Series"
    assert len(forecast) == HORIZON, f"Forecast length should be equal to horizon {HORIZON}"
    assert forecast.dtype == float, "Forecast series should contain float values"
    assert isinstance(forecast.index, pd.DatetimeIndex), "Forecast series index should be DatetimeIndex"
    expected_forecast_start_date = sample_daily_series.index[-1] + pd.Timedelta(days=1)
    assert forecast.index[0] == expected_forecast_start_date, "Forecast index does not start correctly."


def test_exp_smoothing_short_series(short_daily_series):
    """Test Exponential Smoothing with a very short series."""
    # Exponential smoothing might handle shorter series differently than ARIMA
    # Depending on implementation, it might produce a forecast or return None if too short for any model.
    # The current service logic returns (original, None) if too short for seasonal, and tries non-seasonal.
    # If even shorter (e.g. <2 points), it returns (original, None).
    original_series, forecast = forecast_exponential_smoothing(short_daily_series.copy(), HORIZON)
    assert_series_equal(original_series, short_daily_series)
    
    # For 5 points, non-seasonal Holt's method (trend='add') should work.
    if len(short_daily_series) < 2: # As per service logic
        assert forecast is None, "Forecast should be None for series < 2 points"
    else:
        assert isinstance(forecast, pd.Series), "Forecast should be a Series for 5 points (non-seasonal)"
        assert len(forecast) == HORIZON

def test_exp_smoothing_very_short_series_len_1():
    """Test Exponential Smoothing with only 1 data point."""
    single_point_series = pd.Series([100.0], index=pd.to_datetime(['2023-01-01']))
    original_series, forecast = forecast_exponential_smoothing(single_point_series.copy(), HORIZON)
    assert_series_equal(original_series, single_point_series)
    assert forecast is None, "Forecast should be None for a single data point series"


def test_exp_smoothing_monthly_data_with_seasonality(sample_monthly_series):
    """Test Exponential Smoothing with monthly data that has seasonality."""
    # sample_monthly_series is 36 periods long, with freq='MS'
    # This should be enough for seasonal_periods=12
    original_series, forecast = forecast_exponential_smoothing(sample_monthly_series.copy(), HORIZON)
    
    assert isinstance(forecast, pd.Series), "Forecast should be generated for monthly seasonal data"
    assert len(forecast) == HORIZON
    assert forecast.index.freq == sample_monthly_series.index.freq, "Forecast frequency should match input"
    # Check if forecast index starts correctly after the last date of original series
    # For 'MS' freq, adding 1 month to last date is not direct like daily. pd.date_range handles this.
    expected_start = (sample_monthly_series.index[-1] + pd.offsets.MonthBegin(1))
    assert forecast.index[0] == expected_start


# --- Generic tests for all models in AVAILABLE_MODELS ---
@pytest.mark.parametrize("model_name", AVAILABLE_MODELS.keys())
def test_all_models_handle_standard_input(model_name, sample_daily_series):
    """Generic test for all registered models with standard input."""
    forecasting_function = AVAILABLE_MODELS[model_name]
    original_series, forecast = forecasting_function(sample_daily_series.copy(), HORIZON)

    assert isinstance(original_series, pd.Series)
    assert_series_equal(original_series, sample_daily_series)
    assert isinstance(forecast, pd.Series)
    assert len(forecast) == HORIZON
    assert forecast.dtype == float
    assert isinstance(forecast.index, pd.DatetimeIndex)

@pytest.mark.parametrize("model_name", AVAILABLE_MODELS.keys())
def test_all_models_handle_invalid_series_type(model_name):
    """Test models with completely invalid series type (e.g., a list)."""
    # The services themselves now expect pandas Series and handle index conversion.
    # If a raw list is passed, it would fail early unless the service explicitly handles it.
    # Current implementation of forecasting functions try to convert index to pd.DatetimeIndex.
    # If a plain list is passed, it has no index.
    # Let's assume the app.py layer correctly prepares a pd.Series with pd.DatetimeIndex.
    # This test is more about internal robustness if somehow a non-datetime-indexed series is passed.
    
    # Test with a series that has a non-datetime index
    non_datetime_indexed_series = pd.Series([1,2,3,4,5,6,7,8,9,10])
    original_series, forecast = AVAILABLE_MODELS[model_name](non_datetime_indexed_series, HORIZON)
    # Expecting functions to return (None, None) or (original, None) if they can't convert index
    if forecast is not None: # Some models might be more robust or have different fallbacks
        print(f"Model {model_name} produced a forecast for non-datetime index. Output: {forecast}")
    # For this test, the primary check is that it doesn't raise an unhandled exception.
    # The functions are designed to return (original, None) or (None, None) on such errors.
    assert forecast is None or isinstance(forecast, pd.Series)


# Instructions for running tests:
# 1. Ensure `pytest` is installed (it's in requirements.txt).
# 2. Navigate to the project root directory in your terminal.
# 3. Run the command: `pytest`
#    Pytest will automatically discover and run tests in files named `test_*.py` or `*_test.py`.
#
# To run tests for a specific file:
#    `pytest tests/test_forecasting_service.py`
#
# To run a specific test function (e.g., by name):
#    `pytest -k "test_arima_returns_series_and_forecast"`
#
# Ensure your PYTHONPATH is set up if you encounter import errors. If `services` is a top-level
# directory and `tests` is also top-level, running `pytest` from the project root usually works.
# If not, you might need to add the project root to PYTHONPATH:
#    `export PYTHONPATH=$PYTHONPATH:.` (Linux/macOS)
#    `set PYTHONPATH=%PYTHONPATH%;.` (Windows)
#
# The tests assume that the `services.forecasting_service` module and its dependencies
# (pandas, statsmodels) are correctly installed and accessible.

# Note: Some tests, especially for ARIMA, can be slow. For faster feedback during development,
# you might comment out slower tests or use pytest markers to skip them.
# e.g. `@pytest.mark.slow` and then run `pytest -m "not slow"`.
# For this project, the dataset sizes are small, so it should be acceptable.
