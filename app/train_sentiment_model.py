import json
import boto3
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import onnx
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType

# Load data from S3
s3_bucket = "crypto-sentiment-bucket2"
s3_file_key = "sentiment.json"

s3 = boto3.client("s3")
obj = s3.get_object(Bucket=s3_bucket, Key=s3_file_key)
data = obj["Body"].read().decode("utf-8")

df = pd.read_json(StringIO(data))

# Encode Sentiment Labels
label_mapping = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0, "MIXED": 2}
df["sentiment_label"] = df["sentiment_label"].map(label_mapping)

# Convert Messages to TF-IDF Features
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["message"])
y = df["sentiment_label"]

# Split Data into Train & Validation Sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the Model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model Accuracy
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save Model & Vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Convert Model to ONNX Format
initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type)
with open("sentiment_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model trained & saved as ONNX: sentiment_model.onnx")

