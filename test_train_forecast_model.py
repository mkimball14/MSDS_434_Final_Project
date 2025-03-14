import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime, timedelta

def test_train_forecast_model():
    # Define models directory
    MODELS_DIR = os.getenv("MODELS_DIR", "models")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print(f"Models will be saved to {MODELS_DIR}")
    
    # Create sample data similar to your sentiment.json structure
    print("Creating sample data...")
    num_samples = 100
    base_date = datetime(2023, 1, 1)
    
    # Generate random dates over a 60-day period
    dates = [base_date + timedelta(days=np.random.randint(0, 60)) for _ in range(num_samples)]
    dates.sort()  # Sort the dates
    
    # Format dates to match your JSON format
    formatted_dates = [d.strftime("%m.%d.%Y %H:%M:%S") for d in dates]
    
    # Create sample sentiment scores
    sentiment_labels = ["POSITIVE", "NEUTRAL", "NEGATIVE", "MIXED"]
    sample_data = []
    
    for i in range(num_samples):
        # Random sentiment
        sentiment = np.random.choice(sentiment_labels, p=[0.3, 0.4, 0.2, 0.1])
        
        # Generate sentiment scores based on the label
        if sentiment == "POSITIVE":
            pos = np.random.uniform(0.6, 0.9)
            neg = np.random.uniform(0.0, 0.1)
        elif sentiment == "NEGATIVE":
            pos = np.random.uniform(0.0, 0.1)
            neg = np.random.uniform(0.6, 0.9)
        elif sentiment == "NEUTRAL":
            pos = np.random.uniform(0.1, 0.3)
            neg = np.random.uniform(0.1, 0.3)
        else:  # MIXED
            pos = np.random.uniform(0.4, 0.6)
            neg = np.random.uniform(0.4, 0.6)
        
        neutral = max(0, 1.0 - pos - neg - 0.1)
        mixed = max(0, 1.0 - pos - neg - neutral)
        
        sample_data.append({
            "id": i+10000,
            "date": formatted_dates[i],
            "message": f"Sample message {i}",
            "channel_id": "1146170349",
            "user_id": np.random.randint(1000000000, 1999999999),
            "sentiment_label": sentiment,
            "sentiment_score_positive": str(pos),
            "sentiment_score_negative": str(neg),
            "sentiment_score_neutral": str(neutral),
            "sentiment_score_mixed": str(mixed)
        })
    
    # Create dataframe
    df = pd.DataFrame(sample_data)
    
    print(f"Created sample data with {len(df)} records")
    print(df.head())
    
    # Convert date strings to datetime objects
    df["timestamp"] = pd.to_datetime(df["date"], format="%m.%d.%Y %H:%M:%S")
    
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
    
    # Rename count column
    daily_sentiment.rename(columns={"id": "message_count"}, inplace=True)
    
    # Remove days with no data
    daily_sentiment = daily_sentiment.dropna()
    
    print(f"Created time series with {len(daily_sentiment)} days of data")
    
    # Prepare dataframe for Prophet (requires 'ds' and 'y' columns)
    forecast_df = pd.DataFrame({
        'ds': daily_sentiment.index,
        'y': daily_sentiment['compound_score'],
        'message_volume': daily_sentiment['message_count']
    })
    
    # Add additional regressor for message volume
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    model.add_regressor('message_volume')
    
    print("Training forecast model...")
    model.fit(forecast_df)
    print("Model training complete!")
    
    # Generate future dataframe for predictions (next 30 days)
    future = model.make_future_dataframe(periods=30)
    # Add message volume to future
    avg_volume = daily_sentiment['message_count'].mean()
    future['message_volume'] = avg_volume
    
    forecast = model.predict(future)
    
    # Plot forecast
    fig = model.plot(forecast)
    plt.title('Crypto Sentiment Forecast (Test)')
    plt.ylabel('Sentiment Score (Positive - Negative)')
    plt.savefig(os.path.join(MODELS_DIR, 'sentiment_forecast.png'))
    
    # Save components (trend, seasonality)
    fig2 = model.plot_components(forecast)
    fig2.savefig(os.path.join(MODELS_DIR, 'sentiment_components.png'))
    
    # Save model and metadata
    model_path = os.path.join(MODELS_DIR, 'sentiment_forecast_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Also save the message volume for inference
    avg_volume_path = os.path.join(MODELS_DIR, 'avg_message_volume.txt')
    with open(avg_volume_path, 'w') as f:
        f.write(str(avg_volume))
    
    # Save metadata about the model
    metadata = {
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_points": len(forecast_df),
        "forecast_days": 30,
        "avg_message_volume": float(avg_volume),
    }
    
    metadata_path = os.path.join(MODELS_DIR, 'forecast_model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Test forecast model training and artifact creation complete!")
    return model_path

if __name__ == "__main__":
    try:
        model_path = test_train_forecast_model()
        print(f"SUCCESS: Test model created at {model_path}")
    except Exception as e:
        print(f"ERROR: Test failed - {str(e)}")
        raise 