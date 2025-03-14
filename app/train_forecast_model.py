import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import boto3
from io import StringIO
import json
import os
from datetime import datetime

def augment_sparse_data(daily_sentiment, min_days=14):
    """
    Augment sparse time series data to ensure minimum days of data.
    Uses interpolation for missing days and smoothing for noise reduction.
    """
    if len(daily_sentiment) >= min_days:
        return daily_sentiment
    
    print(f"Augmenting sparse dataset ({len(daily_sentiment)} days) to ensure minimum {min_days} days")
    
    # Get date range
    start_date = daily_sentiment.index.min()
    end_date = daily_sentiment.index.max()
    
    # If date range is too short, extend it
    if (end_date - start_date).days < min_days - 1:
        days_to_add = min_days - 1 - (end_date - start_date).days
        end_date = end_date + pd.Timedelta(days=days_to_add)
    
    # Create full date range with all days
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Reindex and interpolate
    augmented_data = daily_sentiment.reindex(full_date_range)
    
    # Linear interpolation for missing values
    augmented_data = augmented_data.interpolate(method='linear')
    
    # For remaining NaNs (at the edges), use forward/backward fill
    augmented_data = augmented_data.fillna(method='ffill').fillna(method='bfill')
    
    print(f"Augmented dataset now has {len(augmented_data)} days")
    return augmented_data

def train_forecast_model():
    # Define models directory
    MODELS_DIR = os.getenv("MODELS_DIR", "models")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print(f"Models will be saved to {MODELS_DIR}")
    
    # Load data from S3
    s3_bucket = "crypto-sentiment-bucket2"
    s3_file_key = "sentiment.json"
    
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=s3_bucket, Key=s3_file_key)
    data = obj["Body"].read().decode("utf-8")
    
    # Parse JSON data
    sentiment_data = json.loads(data)
    df = pd.DataFrame(sentiment_data)
    
    print(f"Loaded {len(df)} records from S3")
    
    # Convert date strings to datetime objects
    df["timestamp"] = pd.to_datetime(df["date"], format="%m.%d.%Y %H:%M:%S")
    
    # Create numeric sentiment values
    # We'll use the actual sentiment scores instead of just labels
    df["sentiment_score_positive"] = df["sentiment_score_positive"].astype(float)
    df["sentiment_score_negative"] = df["sentiment_score_negative"].astype(float)
    df["sentiment_score_neutral"] = df["sentiment_score_neutral"].astype(float)
    df["sentiment_score_mixed"] = df["sentiment_score_mixed"].astype(float)
    
    # Calculate a compound sentiment score (-1 to 1 scale)
    # Higher value = more positive, lower value = more negative
    df["compound_score"] = df["sentiment_score_positive"] - df["sentiment_score_negative"]
    
    # Aggregate sentiment by day
    # Group by day and calculate average sentiment
    daily_sentiment = df.groupby(pd.Grouper(key="timestamp", freq="D")).agg({
        "compound_score": "mean",
        "sentiment_score_positive": "mean",
        "sentiment_score_negative": "mean",
        "sentiment_score_neutral": "mean",
        "sentiment_score_mixed": "mean",
        "id": "count"  # Count messages per day
    })
    
    # Rename count column
    daily_sentiment.rename(columns={"id": "message_count"}, inplace=True)
    
    # After creating daily_sentiment and before removing NaNs
    if len(daily_sentiment.dropna()) < 14:
        print("Dataset too sparse, attempting augmentation")
        daily_sentiment = augment_sparse_data(daily_sentiment)
    else:
        # Only remove NaNs if we have enough data
        daily_sentiment = daily_sentiment.dropna()
    
    print(f"Created time series with {len(daily_sentiment)} days of data")
    
    # Ensure we have enough data
    if len(daily_sentiment) < 30:
        print("Warning: Less than 30 days of data available for forecasting")
        if len(daily_sentiment) < 7:  # Reduce minimum requirement from 14 to 7 days
            raise ValueError("Not enough data for forecasting, need at least 7 days")
    
    # Prepare dataframe for Prophet (requires 'ds' and 'y' columns)
    forecast_df = pd.DataFrame({
        'ds': daily_sentiment.index,
        'y': daily_sentiment['compound_score'],
        'message_volume': daily_sentiment['message_count']
    })
    
    # Adapt model parameters for small datasets
    use_yearly_seasonality = len(forecast_df) >= 365
    use_weekly_seasonality = len(forecast_df) >= 14
    use_daily_seasonality = len(forecast_df) >= 7

    # Add additional regressor for message volume
    model = Prophet(
        daily_seasonality=use_daily_seasonality,
        weekly_seasonality=use_weekly_seasonality,
        yearly_seasonality=use_yearly_seasonality,
        seasonality_mode='additive',  # More stable for small datasets
        uncertainty_samples=100       # Reduce computation time
    )
    model.add_regressor('message_volume')
    
    print("Training forecast model...")
    model.fit(forecast_df)
    print("Model training complete!")
    
    # Calculate appropriate forecast period based on data amount
    # Generally, don't forecast more than 50% of your historical data length
    forecast_days = min(30, len(daily_sentiment) // 2)
    if forecast_days < 3:
        forecast_days = 3  # Minimum forecast period

    print(f"Using forecast period of {forecast_days} days based on available data")

    # Generate future dataframe for predictions
    future = model.make_future_dataframe(periods=forecast_days)
    # Add message volume to future
    # For simplicity, we'll use the average message volume from the last 7 days
    avg_volume = daily_sentiment['message_count'].tail(7).mean()
    future['message_volume'] = avg_volume
    
    forecast = model.predict(future)
    
    # Plot forecast
    fig = model.plot(forecast)
    plt.title('Crypto Sentiment Forecast')
    plt.ylabel('Sentiment Score (Positive - Negative)')
    plt.savefig(os.path.join(MODELS_DIR, 'sentiment_forecast.png'))
    
    # Save components (trend, seasonality)
    fig2 = model.plot_components(forecast)
    fig2.savefig(os.path.join(MODELS_DIR, 'sentiment_components.png'))
    
    # Save model and metadata
    model_path = os.path.join(MODELS_DIR, 'sentiment_forecast_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Also save the last 7 days' average message volume for inference
    avg_volume_path = os.path.join(MODELS_DIR, 'avg_message_volume.txt')
    with open(avg_volume_path, 'w') as f:
        f.write(str(avg_volume))
    
    # Save metadata about the model
    metadata = {
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_points": len(forecast_df),
        "forecast_days": forecast_days,
        "avg_message_volume": avg_volume,
    }
    
    metadata_path = os.path.join(MODELS_DIR, 'forecast_model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Evaluate on test data
    # For simplicity, we'll use the last 14 days as test data (if we have enough data)
    test_size = min(14, len(forecast_df) // 3)  # Use at most 1/3 of data for testing
    
    if len(forecast_df) > test_size + 14:  # Need at least 14 days for training
        train = forecast_df[:-test_size]
        test = forecast_df[-test_size:]
        
        model_test = Prophet(
            daily_seasonality=use_daily_seasonality,
            weekly_seasonality=use_weekly_seasonality,
            yearly_seasonality=use_yearly_seasonality,
            seasonality_mode='additive',
            uncertainty_samples=100
        )
        model_test.add_regressor('message_volume')
        model_test.fit(train)
        
        test_future = model_test.make_future_dataframe(periods=test_size)
        test_future['message_volume'] = test['message_volume'].values
        test_forecast = model_test.predict(test_future)
        
        # Calculate MAE on test set
        test_predictions = test_forecast[-test_size:]['yhat'].values
        test_actual = test['y'].values
        mae = mean_absolute_error(test_actual, test_predictions)
        print(f"Forecast Model MAE: {mae:.4f}")
        
        # Plot test results
        plt.figure(figsize=(12, 6))
        plt.plot(test['ds'], test_actual, 'b-', label='Actual')
        plt.plot(test['ds'], test_predictions, 'r--', label='Predicted')
        plt.legend()
        plt.title('Sentiment Forecast Model Validation')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.savefig(os.path.join(MODELS_DIR, 'forecast_validation.png'))
        
        # Save evaluation metrics
        metrics = {
            "mean_absolute_error": float(mae),
            "test_size": test_size
        }
        metrics_path = os.path.join(MODELS_DIR, 'forecast_model_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    print("Forecast model training and evaluation complete!")
    
    # Return metrics and insights
    return {
        "model_file": model_path,
        "forecast_days": forecast_days,
        "data_points_used": len(forecast_df),
        "avg_sentiment": forecast_df['y'].mean(),
        "sentiment_trend": "Upward" if forecast['trend'].tail(forecast_days).mean() > forecast['trend'].head(forecast_days).mean() else "Downward",
        "forecast_plots": [
            os.path.join(MODELS_DIR, "sentiment_forecast.png"), 
            os.path.join(MODELS_DIR, "sentiment_components.png")
        ]
    }

if __name__ == "__main__":
    try:
        result = train_forecast_model()
        print("Forecast model training successful!")
        print(f"Model saved to {result['model_file']}")
    except Exception as e:
        print(f"Error training forecast model: {str(e)}")
        raise 