import os
import glob
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

print("Starting model training...")

# --- Define Paths ---
# MODIFICATION: Removed ".." from the paths.
# We are running from the project root, so paths are relative to it.
data_dir = os.path.join("output", "processed_data")
model_path = os.path.join("output", "models")
os.makedirs(model_path, exist_ok=True)

def find_spark_csv(folder_path):
    """Finds the CSV file inside a folder saved by Spark."""
    # MODIFICATION: Changed glob pattern to be more specific to 'part-' files
    # This avoids accidentally grabbing other .csv files if they exist.
    search_path = os.path.join(folder_path, "part-*.csv")
    try:
        # Find all files matching the pattern
        csv_files = glob.glob(search_path)
        if not csv_files:
            # Raise IndexError if list is empty, to be caught below
            raise IndexError
            
        # Select the first file found
        csv_file_path = csv_files[0]
        print(f"Found data file: {csv_file_path}")
        return pd.read_csv(csv_file_path)
        
    except IndexError:
        print(f"Error: No 'part-*.csv' file found in {folder_path}.")
        print("Did 'preprocessing.py' run successfully?")
        return None

# 1. Load Processed Data
print("Loading processed training and validation data...")
df_train = find_spark_csv(os.path.join(data_dir, "train_processed"))
df_test = find_spark_csv(os.path.join(data_dir, "validation_processed"))

if df_train is None or df_test is None:
    print("Exiting. Preprocessed data not found.")
    exit()

# Handle potential nulls that slipped through
df_train = df_train.dropna(subset=['final_text', 'sentiment'])
df_test = df_test.dropna(subset=['final_text', 'sentiment'])

# Ensure sentiment labels match (e.g., 'Positive' -> 'positive')
df_train['sentiment'] = df_train['sentiment'].str.lower()
df_test['sentiment'] = df_test['sentiment'].str.lower()

# Filter out any remaining 'irrelevant'
df_train = df_train[df_train['sentiment'] != 'irrelevant']
df_test = df_test[df_test['sentiment'] != 'irrelevant']


print(f"Loaded {len(df_train)} training tweets and {len(df_test)} test tweets.")

# 2. Define X and y (NO TRAIN_TEST_SPLIT NEEDED)
X_train = df_train['final_text'].astype(str)
y_train = df_train['sentiment']
X_test = df_test['final_text'].astype(str)
y_test = df_test['sentiment']

# 3. Vectorize Text
print("Vectorizing text...")
# Fit the vectorizer ONLY on the training data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
# Transform the test data using the *same* fitted vectorizer
X_test_vec = vectorizer.transform(X_test)

# 4. Train Model
print("Training Logistic Regression model...")
model = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=1000, random_state=42)
# Fit the model ONLY on the training data
model.fit(X_train_vec, y_train)

print("Model training complete.")

# 5. Evaluate Model
# Evaluate the model on the unseen test (validation) data
print("Evaluating model on test set...")
y_pred = model.predict(X_test_vec)

print("\n--- Classification Report (on validation data) ---")
print(classification_report(y_test, y_pred, zero_division=0))
print("--------------------------------------------------")

# 6. Save Model and Vectorizer
joblib.dump(model, os.path.join(model_path, "sentiment_model.pkl"))
joblib.dump(vectorizer, os.path.join(model_path, "tfidf_vectorizer.pkl"))

print(f"Model and vectorizer saved to {model_path}")