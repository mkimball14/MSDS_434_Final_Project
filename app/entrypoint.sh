#!/bin/bash
set -e

echo "Starting service initialization..."

# Set models directory
export MODELS_DIR=${MODELS_DIR:-"models"}
echo "Models directory set to: $MODELS_DIR"
mkdir -p $MODELS_DIR

# Try to train the forecast model
echo "Attempting to train forecast model..."
python train_forecast_model.py || echo "Forecast model training failed, proceeding without it"

# Start the API service
echo "Starting API service..."
exec uvicorn app:app --host 0.0.0.0 --port 8080 