from fastapi import FastAPI, HTTPException
import onnxruntime as ort
import numpy as np
import joblib
import boto3
import os
from io import StringIO
import pandas as pd
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from typing import List, Dict, Optional
from fastapi.responses import HTMLResponse
import matplotlib.pyplot as plt
from datetime import datetime
import io
import base64
import json

# Define paths
MODELS_DIR = os.getenv("MODELS_DIR", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Load S3 bucket name from environment variables
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "crypto-sentiment-bucket2")
S3_FILE_KEY = "sentiment.json"

# Initialize AWS S3 client
s3 = boto3.client("s3")

# Model file paths
onnx_model_path = os.path.join(MODELS_DIR, "sentiment_model.onnx")
vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
forecast_model_path = os.path.join(MODELS_DIR, "sentiment_forecast_model.pkl")
avg_message_volume_path = os.path.join(MODELS_DIR, "avg_message_volume.txt")

# Load ONNX model & vectorizer
onnx_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
vectorizer = joblib.load(vectorizer_path)

# Try to load the forecast model if it exists
try:
    from prophet import Prophet
    
    if os.path.exists(forecast_model_path):
        forecast_model = joblib.load(forecast_model_path)
        
        if os.path.exists(avg_message_volume_path):
            with open(avg_message_volume_path, 'r') as f:
                avg_message_volume = float(f.read().strip())
        else:
            # Fallback to default if file doesn't exist
            avg_message_volume = 100.0
            
        forecast_available = True
        print(f"Forecast model loaded successfully from {forecast_model_path}")
    else:
        # Try loading from current directory (for backward compatibility)
        try:
            forecast_model = joblib.load("sentiment_forecast_model.pkl")
            with open('avg_message_volume.txt', 'r') as f:
                avg_message_volume = float(f.read().strip())
            forecast_available = True
            print("Forecast model loaded successfully from current directory")
        except:
            forecast_available = False
            print("Forecast model not found")
except Exception as e:
    forecast_available = False
    print(f"Forecast model not available: {str(e)}")

# Load forecast model metadata if available
forecast_metadata = {}
metadata_path = os.path.join(MODELS_DIR, "forecast_model_metadata.json")
if os.path.exists(metadata_path):
    try:
        with open(metadata_path, 'r') as f:
            forecast_metadata = json.load(f)
        print(f"Loaded forecast metadata: {forecast_metadata}")
    except Exception as e:
        print(f"Error loading forecast metadata: {str(e)}")

# Maximum forecast days - will be overridden by metadata if available
MAX_FORECAST_DAYS = forecast_metadata.get("forecast_days", 30)

# Define sentiment labels
sentiment_map = {1: "POSITIVE", 0: "NEUTRAL", -1: "NEGATIVE", 2: "MIXED"}

# API Initialization
app = FastAPI()

Instrumentator().instrument(app).expose(app)

class SentimentRequest(BaseModel):
    message: str

@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    try:
        input_text = [request.message]
        input_vector = vectorizer.transform(input_text).toarray().astype(np.float32)

        # Run ONNX inference
        result = onnx_session.run(None, {"input": input_vector})[0][0]
        predicted_label = sentiment_map.get(int(result), "UNKNOWN")

        return {"message": request.message, "sentiment": predicted_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Crypto Sentiment Analysis API is Running"}

# Forecasting functionality
if forecast_available:
    class ForecastRequest(BaseModel):
        days_ahead: int = 7
        message_volume_multiplier: Optional[float] = 1.0  # To simulate higher/lower activity

    class ForecastResponse(BaseModel):
        dates: List[str]
        sentiment_scores: List[float] 
        confidence_intervals: List[Dict[str, float]]
        trend: List[float]
        weekday_effect: List[float]

    @app.post("/forecast")
    def forecast_sentiment(request: ForecastRequest):
        try:
            if request.days_ahead < 1:
                raise HTTPException(status_code=400, detail="Days ahead must be at least 1")
            
            # Limit forecast days to what the model supports
            if request.days_ahead > MAX_FORECAST_DAYS:
                print(f"Requested {request.days_ahead} days but model only supports {MAX_FORECAST_DAYS} days")
                request.days_ahead = MAX_FORECAST_DAYS
            
            # Generate future dataframe for predictions
            future = forecast_model.make_future_dataframe(periods=request.days_ahead)
            
            # Set the message volume
            projected_volume = avg_message_volume * request.message_volume_multiplier
            future['message_volume'] = projected_volume
            
            # Make prediction
            forecast = forecast_model.predict(future)
            
            # Extract the forecasted values for the requested days
            forecast_result = forecast.tail(request.days_ahead)
            
            # Format the response
            dates = [d.strftime("%Y-%m-%d") for d in forecast_result["ds"]]
            
            return {
                "dates": dates,
                "sentiment_scores": forecast_result["yhat"].tolist(),
                "confidence_intervals": [
                    {"lower": float(lower), "upper": float(upper)} 
                    for lower, upper in zip(forecast_result["yhat_lower"], forecast_result["yhat_upper"])
                ],
                "trend": forecast_result["trend"].tolist(),
                "weekday_effect": forecast_result["weekly"].tolist()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/forecast_chart/{days}", response_class=HTMLResponse)
    async def get_forecast_chart(days: int = 30):
        if days < 1 or days > 90:
            raise HTTPException(status_code=400, detail="Days ahead must be between 1 and 90")
        
        # Generate future dataframe for predictions
        future = forecast_model.make_future_dataframe(periods=days)
        future['message_volume'] = avg_message_volume
        forecast = forecast_model.predict(future)
        
        # Create a chart using matplotlib
        plt.figure(figsize=(12, 6))
        
        # Plot actual data points
        forecast_df = pd.DataFrame({
            'ds': future['ds'],
            'yhat': forecast['yhat'],
            'yhat_lower': forecast['yhat_lower'],
            'yhat_upper': forecast['yhat_upper'],
        })
        
        # Get today's date
        today = pd.Timestamp.now().normalize()
        
        # Split into historical and future
        historical = forecast_df[forecast_df['ds'] < today]
        future_data = forecast_df[forecast_df['ds'] >= today]
        
        # Plot
        plt.plot(historical['ds'], historical['yhat'], 'b-', label='Historical')
        plt.plot(future_data['ds'], future_data['yhat'], 'r-', label='Forecast')
        plt.fill_between(future_data['ds'], future_data['yhat_lower'], future_data['yhat_upper'], 
                        color='r', alpha=0.2, label='95% Confidence Interval')
        
        plt.axvline(x=today, color='k', linestyle='--', label='Today')
        plt.legend()
        plt.title(f'Cryptocurrency Sentiment Forecast - Next {days} Days')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score (Higher = More Positive)')
        plt.grid(True, alpha=0.3)
        
        # Convert plot to base64 image to embed in HTML
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Create HTML response
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crypto Sentiment Forecast</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                }}
                .container {{
                    max-width: 1000px;
                    margin: 0 auto;
                }}
                .chart {{
                    width: 100%;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    margin: 20px 0;
                }}
                h1 {{
                    color: #333;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Cryptocurrency Sentiment Forecast</h1>
                <p>Showing forecast for the next {days} days</p>
                <img src="data:image/png;base64,{img_str}" class="chart">
                <div>
                    <p><strong>Interpretation:</strong></p>
                    <ul>
                        <li>Higher values indicate more positive sentiment</li>
                        <li>Lower values indicate more negative sentiment</li>
                        <li>The shaded area represents the 95% confidence interval</li>
                    </ul>
                    <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
else:
    @app.get("/forecast_status")
    def forecast_status():
        return {"status": "Forecast model not available", "available": False}

