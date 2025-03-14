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
from matplotlib.dates import DayLocator, DateFormatter
from datetime import datetime
import io
import base64
import json
import logging

# Set up logging at the top of the file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.info(f"Loading forecast model from {forecast_model_path}")
        forecast_model = joblib.load(forecast_model_path)
        
        if os.path.exists(avg_message_volume_path):
            logger.info(f"Loading average message volume from {avg_message_volume_path}")
            with open(avg_message_volume_path, 'r') as f:
                avg_message_volume = float(f.read().strip())
            logger.info(f"Average message volume: {avg_message_volume}")
        else:
            logger.warning("Average message volume file not found, using default")
            avg_message_volume = 100.0
            
        forecast_available = True
        logger.info("Forecast model loaded successfully")
    else:
        logger.warning(f"Forecast model not found at {forecast_model_path}")
        forecast_available = False
except Exception as e:
    forecast_available = False
    logger.error(f"Error loading forecast model: {str(e)}")

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
        logger.info(f"Generating forecast chart for {days} days")
        try:
            if days < 1 or days > 90:
                raise HTTPException(status_code=400, detail="Days ahead must be between 1 and 90")
            
            # Load metadata to get actual data period
            metadata_path = os.path.join(MODELS_DIR, "forecast_model_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Get the actual data range
            earliest_date = pd.to_datetime(metadata["earliest_date"])
            latest_date = pd.to_datetime(metadata["latest_date"])
            forecast_days = metadata["forecast_days"]
            
            # Generate future dataframe for predictions
            logger.info("Creating future dataframe")
            future = forecast_model.make_future_dataframe(periods=forecast_days)
            future['message_volume'] = avg_message_volume
            logger.info("Making forecast predictions")
            forecast = forecast_model.predict(future)
            logger.info("Forecast predictions completed")
            
            # Create a chart using matplotlib
            logger.info("Creating visualization")
            plt.figure(figsize=(12, 6))
            
            # Plot actual data points
            forecast_df = pd.DataFrame({
                'ds': future['ds'],
                'yhat': forecast['yhat'],
                'yhat_lower': forecast['yhat_lower'],
                'yhat_upper': forecast['yhat_upper'],
            })
            
            # Split into historical and future based on actual data range
            historical = forecast_df[forecast_df['ds'] <= latest_date]
            future_data = forecast_df[forecast_df['ds'] > latest_date]
            
            # Plot with actual data range
            plt.plot(historical['ds'], historical['yhat'], 'b-', label='Historical', linewidth=2, markersize=6)
            plt.plot(future_data['ds'], future_data['yhat'], 'r-', label='Forecast', linewidth=2, markersize=6)
            plt.fill_between(future_data['ds'], future_data['yhat_lower'], future_data['yhat_upper'], 
                            color='r', alpha=0.2, label='95% Confidence Interval')
            
            plt.axvline(x=latest_date, color='k', linestyle='--', label='Last Historical Date')
            plt.legend()
            plt.title('Cryptocurrency Sentiment Forecast\nJanuary 2021 Data', fontsize=12, pad=20)
            plt.xlabel('Date')
            plt.ylabel('Sentiment Score (Higher = More Positive)')
            plt.grid(True, alpha=0.3)
            
            # Format x-axis to show daily ticks
            plt.gca().xaxis.set_major_locator(DayLocator(interval=1))
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            
            # Set x-axis limits to focus on the actual data period
            plt.xlim(earliest_date - pd.Timedelta(days=1), 
                    latest_date + pd.Timedelta(days=forecast_days + 1))
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add text box with data summary
            text = f'Data Summary:\n' \
                   f'Period: {earliest_date.strftime("%Y-%m-%d")} to {latest_date.strftime("%Y-%m-%d")}\n' \
                   f'Messages per day: {int(avg_message_volume)}'
            plt.text(0.02, 0.98, text, transform=plt.gca().transAxes, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                     verticalalignment='top', fontsize=8)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Convert plot to base64 image to embed in HTML
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
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
                    <p>Showing forecast for the next {forecast_days} days</p>
                    <img src="data:image/png;base64,{img_str}" class="chart">
                    <div>
                        <p><strong>Interpretation:</strong></p>
                        <ul>
                            <li>Higher values indicate more positive sentiment</li>
                            <li>Lower values indicate more negative sentiment</li>
                            <li>The shaded area represents the 95% confidence interval</li>
                        </ul>
                        <p>Last updated: {metadata["training_date"]}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            return html_content
        except Exception as e:
            logger.error(f"Error generating forecast chart: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
else:
    @app.get("/forecast_status")
    def forecast_status():
        return {"status": "Forecast model not available", "available": False}


