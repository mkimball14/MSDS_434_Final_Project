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
    
    print("\nAttempting to load data from S3...")
    print(f"Bucket: {s3_bucket}")
    print(f"Key: {s3_file_key}")
    
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=s3_bucket, Key=s3_file_key)
    data = obj["Body"].read().decode("utf-8")
    
    # Print first few records of raw data
    print("\nFirst few records of raw data:")
    raw_data = json.loads(data)
    for i, record in enumerate(raw_data[:2]):
        print(f"\nRecord {i + 1}:")
        print(json.dumps(record, indent=2))
    
    # Parse JSON data
    sentiment_data = json.loads(data)
    df = pd.DataFrame(sentiment_data)
    
    print(f"\nLoaded {len(df)} records from S3")
    print("\nDataFrame columns:", df.columns.tolist())
    print("\nFirst few dates before parsing:")
    print(df["date"].head())
    
    # Convert date strings to datetime objects - handle multiple possible formats
    def parse_date(date_str):
        try:
            # Try multiple date formats
            for fmt in ["%Y-%m-%d %H:%M:%S", "%m.%d.%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
                try:
                    dt = pd.to_datetime(date_str, format=fmt)
                    print(f"Successfully parsed date {date_str} with format {fmt}")
                    return dt
                except:
                    continue
            # If none of the specific formats work, try the default parser
            print(f"Trying default parser for date: {date_str}")
            return pd.to_datetime(date_str)
        except Exception as e:
            print(f"Warning: Could not parse date: {date_str}, Error: {str(e)}")
            return None

    print("\nParsing dates...")
    df["timestamp"] = df["date"].apply(parse_date)
    
    # Remove invalid dates
    df = df.dropna(subset=["timestamp"])
    print(f"\nRemaining records after removing invalid dates: {len(df)}")
    
    # Print data summary
    latest_date = df["timestamp"].max()
    earliest_date = df["timestamp"].min()
    
    print(f"\nData summary:")
    print(f"Earliest date: {earliest_date}")
    print(f"Latest date: {latest_date}")
    print(f"Date range: {(latest_date - earliest_date).days} days")
    print("\nSample of parsed dates:")
    print(df[["date", "timestamp"]].head())
    
    # Ensure timestamps are in chronological order
    df = df.sort_values("timestamp")
    
    # Create numeric sentiment values
    df["sentiment_score_positive"] = df["sentiment_score_positive"].astype(float)
    df["sentiment_score_negative"] = df["sentiment_score_negative"].astype(float)
    df["sentiment_score_neutral"] = df["sentiment_score_neutral"].astype(float)
    df["sentiment_score_mixed"] = df["sentiment_score_mixed"].astype(float)
    
    # Calculate a compound sentiment score (-1 to 1 scale)
    df["compound_score"] = df["sentiment_score_positive"] - df["sentiment_score_negative"]
    
    # Aggregate sentiment by day
    daily_sentiment = df.groupby(pd.Grouper(key="timestamp", freq="D")).agg({
        "compound_score": "mean",
        "sentiment_score_positive": "mean",
        "sentiment_score_negative": "mean",
        "sentiment_score_neutral": "mean",
        "sentiment_score_mixed": "mean",
        "id": "count"  # Count messages per day
    })
    
    # Remove days with no data
    daily_sentiment = daily_sentiment[daily_sentiment["id"] > 0]
    
    # Rename count column
    daily_sentiment.rename(columns={"id": "message_count"}, inplace=True)
    
    print(f"\nProcessed data summary:")
    print(f"Date range: {daily_sentiment.index.min()} to {daily_sentiment.index.max()}")
    print(f"Number of days with data: {len(daily_sentiment)}")
    
    # After creating daily_sentiment and before removing NaNs
    if len(daily_sentiment.dropna()) < 14:
        print("Dataset too sparse, attempting augmentation")
        daily_sentiment = augment_sparse_data(daily_sentiment, min_days=30)  # Increase minimum days
    else:
        # Only remove NaNs if we have enough data
        daily_sentiment = daily_sentiment.dropna()
    
    print(f"Created time series with {len(daily_sentiment)} days of data")
    
    # Prepare dataframe for Prophet
    forecast_df = pd.DataFrame({
        'ds': daily_sentiment.index,
        'y': daily_sentiment['compound_score'],
        'message_volume': daily_sentiment['message_count']
    })
    
    # Set up Prophet model parameters for small dataset
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=False,  # Disable weekly seasonality for small dataset
        yearly_seasonality=False,  # 2021 data only
        seasonality_mode='additive',
        uncertainty_samples=1000,  # Increase uncertainty samples
        growth='linear',
        changepoint_prior_scale=0.05,  # Reduce flexibility for small dataset
        seasonality_prior_scale=10.0,  # Increase seasonality strength
    )
    model.add_regressor('message_volume')
    
    print("Training forecast model...")
    model.fit(forecast_df)
    print("Model training complete!")
    
    # Use a 3-day forecast period for small dataset
    forecast_days = 3
    print(f"Using {forecast_days}-day forecast period for small dataset")

    # Generate future dataframe for predictions
    future = model.make_future_dataframe(periods=forecast_days)
    
    # Add message volume to future
    avg_volume = daily_sentiment['message_count'].mean()  # Use overall mean for small dataset
    future['message_volume'] = avg_volume
    
    forecast = model.predict(future)
    
    # Plot forecast
    fig = plt.figure(figsize=(12, 6))
    
    # Plot actual data points with markers
    plt.plot(forecast_df['ds'], forecast_df['y'], 'bo-', label='Historical', linewidth=2, markersize=6)
    
    # Plot forecast
    future_dates = forecast['ds'].tail(forecast_days)
    future_values = forecast['yhat'].tail(forecast_days)
    future_lower = forecast['yhat_lower'].tail(forecast_days)
    future_upper = forecast['yhat_upper'].tail(forecast_days)
    
    plt.plot(future_dates, future_values, 'ro-', label='Forecast', linewidth=2, markersize=6)
    plt.fill_between(future_dates, future_lower, future_upper, color='r', alpha=0.2, label='95% Confidence Interval')
    
    # Add reference line at the end of historical data
    last_historical_date = forecast_df['ds'].max()
    plt.axvline(x=last_historical_date, color='k', linestyle='--', label='Last Historical Date')
    
    plt.legend(loc='best', fontsize=10)
    plt.title('Cryptocurrency Sentiment Forecast\nJanuary 2021 Data', fontsize=12, pad=20)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Sentiment Score\n(Higher = More Positive)', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_locator(plt.DayLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(plt.DateFormatter('%Y-%m-%d'))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add text box with data summary
    text = f'Data Summary:\n' \
           f'Period: {forecast_df["ds"].min().strftime("%Y-%m-%d")} to {forecast_df["ds"].max().strftime("%Y-%m-%d")}\n' \
           f'Messages per day: {int(daily_sentiment["message_count"].mean())}'
    plt.text(0.02, 0.98, text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             verticalalignment='top', fontsize=8)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(os.path.join(MODELS_DIR, 'sentiment_forecast.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
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
        "data_period": "2021",
        "earliest_date": earliest_date.strftime("%Y-%m-%d"),
        "latest_date": latest_date.strftime("%Y-%m-%d")
    }
    
    metadata_path = os.path.join(MODELS_DIR, 'forecast_model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Evaluate on test data
    test_size = min(7, len(forecast_df) // 3)  # Use at most 1/3 of data for testing
    
    if len(forecast_df) > test_size + 14:  # Need at least 14 days for training
        train = forecast_df[:-test_size]
        test = forecast_df[-test_size:]
        
        model_test = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
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
        plt.title('Sentiment Forecast Model Validation (2021 Data)')
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
