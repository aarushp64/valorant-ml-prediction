import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ARIMAForecaster:
    """ARIMA model for time series forecasting of vehicle sales data."""
    
    def __init__(self, order=(1, 1, 1)):
        """
        Initialize the ARIMA forecaster.
        
        Parameters:
        -----------
        order : tuple
            The (p, d, q) order of the ARIMA model.
        """
        self.order = order
        self.model = None
        self.model_fit = None
        
    def fit(self, data):
        """
        Fit the ARIMA model to the time series data.
        
        Parameters:
        -----------
        data : pandas.Series or array-like
            The time series data to fit the model to.
        
        Returns:
        --------
        self : ARIMAForecaster
            The fitted forecaster.
        """
        self.model = ARIMA(data, order=self.order)
        self.model_fit = self.model.fit()
        return self
    
    def predict(self, steps=12):
        """
        Generate forecasts for future time periods.
        
        Parameters:
        -----------
        steps : int
            The number of steps to forecast.
        
        Returns:
        --------
        forecast : pandas.Series
            The forecasted values.
        """
        if self.model_fit is None:
            raise ValueError("Model has not been fit yet. Call fit() first.")
        
        forecast = self.model_fit.forecast(steps=steps)
        return forecast
    
    def evaluate(self, test_data, forecast=None):
        """
        Evaluate the model performance on test data.
        
        Parameters:
        -----------
        test_data : pandas.Series
            The actual values to compare against.
        forecast : pandas.Series, optional
            The forecasted values. If None, will generate forecasts for the length of test_data.
        
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics (RMSE, MAE).
        """
        if forecast is None:
            forecast = self.predict(steps=len(test_data))
        
        # Align indices if using pandas Series
        if isinstance(test_data, pd.Series) and isinstance(forecast, pd.Series):
            common_idx = test_data.index.intersection(forecast.index)
            test_data = test_data.loc[common_idx]
            forecast = forecast.loc[common_idx]
        
        rmse = np.sqrt(mean_squared_error(test_data, forecast))
        mae = mean_absolute_error(test_data, forecast)
        
        return {
            'rmse': rmse,
            'mae': mae
        }
    
    def summary(self):
        """
        Get the summary of the fitted ARIMA model.
        
        Returns:
        --------
        summary : str
            The summary of the fitted model.
        """
        if self.model_fit is None:
            raise ValueError("Model has not been fit yet. Call fit() first.")
        
        return self.model_fit.summary()