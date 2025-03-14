import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import json

def test_load_forecast_model():
    print("Testing model loading and forecasting...")
    
    # Define paths
    MODELS_DIR = "models"
    forecast_model_path = os.path.join(MODELS_DIR, "sentiment_forecast_model.pkl")
    avg_message_volume_path = os.path.join(MODELS_DIR, "avg_message_volume.txt")
    
    # Check if model exists
    if not os.path.exists(forecast_model_path):
        print(f"ERROR: Model not found at {forecast_model_path}")
        return False
    
    try:
        # Load the model
        print(f"Loading forecast model from {forecast_model_path}")
        forecast_model = joblib.load(forecast_model_path)
        
        # Load average message volume
        with open(avg_message_volume_path, 'r') as f:
            avg_message_volume = float(f.read().strip())
        
        print(f"Loaded average message volume: {avg_message_volume}")
        
        # Generate future dataframe for predictions
        days_ahead = 7
        future = forecast_model.make_future_dataframe(periods=days_ahead)
        future['message_volume'] = avg_message_volume
        
        # Make prediction
        print("Making forecast prediction...")
        forecast = forecast_model.predict(future)
        
        # Extract the forecasted values for the requested days
        forecast_result = forecast.tail(days_ahead)
        
        # Format the response
        dates = [d.strftime("%Y-%m-%d") for d in forecast_result["ds"]]
        sentiment_scores = forecast_result["yhat"].tolist()
        
        print("Forecast for the next 7 days:")
        for i, (date, score) in enumerate(zip(dates, sentiment_scores)):
            print(f"  Day {i+1}: {date} - Sentiment score: {score:.4f}")
        
        # Create a test visualization
        plt.figure(figsize=(10, 6))
        plt.plot(forecast_result["ds"], forecast_result["yhat"], 'r-', label='Forecast')
        plt.fill_between(
            forecast_result["ds"], 
            forecast_result["yhat_lower"], 
            forecast_result["yhat_upper"],
            color='r', 
            alpha=0.2, 
            label='95% Confidence Interval'
        )
        plt.title('Cryptocurrency Sentiment 7-Day Forecast')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(MODELS_DIR, 'test_app_forecast.png'))
        
        print(f"Test visualization saved to {os.path.join(MODELS_DIR, 'test_app_forecast.png')}")
        
        print("Model loading and forecasting test PASSED!")
        return True
        
    except Exception as e:
        print(f"ERROR during testing: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_load_forecast_model()
    print(f"Test result: {'SUCCESS' if success else 'FAILURE'}") 