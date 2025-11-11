import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import split, concat_ws

print("Starting Spark preprocessing for train and validation sets...")

# Setup Spark Session
spark = SparkSession.builder \
    .appName("TweetPreprocessing") \
    .master("local[*]") \
    .getOrCreate()

def process_file(input_path, output_path):
    """Loads, cleans, and saves a single tweet CSV file."""
    print(f"Processing {input_path}...")
    
    # 1. Load Data
    try:
        # MODIFICATION: Changed header=True to header=False
        # The CSV files for this dataset do not have a header row.
        df = spark.read.csv(input_path, header=False, inferSchema=True)
    except Exception as e:
        print(f"Error loading data from {input_path}: {e}")
        return

    # Select and rename relevant columns
    # Based on the dataset, the schema is:
    # _c0: Tweet ID
    # _c1: Entity
    # _c2: Sentiment
    # _c3: Text content
    try:
        # MODIFICATION: Removed the old try/except block and replaced with this.
        # We select the default column names and rename them to what
        # the rest of the script expects ("text" and "sentiment").
        df = df.select(
            col("_c3").alias("text"),
            col("_c2").alias("sentiment")
        )
    except Exception as e:
        print(f"Error selecting _c2 or _c3. Are you sure the CSV is correct? {e}")
        return

    # 2. Clean Text
    df = df.withColumn("text", col("text").cast("string"))
    df = df.filter(col("text").isNotNull() & col("sentiment").isNotNull())
    
    # Filter out 'Irrelevant' sentiment if it exists
    df = df.filter(col("sentiment") != "Irrelevant")

    cleaned_df = df.withColumn("cleaned_text", lower(col("text")))
    cleaned_df = cleaned_df.withColumn("cleaned_text", regexp_replace("cleaned_text", r"http\S+|https\S+", ""))
    cleaned_df = cleaned_df.withColumn("cleaned_text", regexp_replace("cleaned_text", r"@\w+", ""))
    cleaned_df = cleaned_df.withColumn("cleaned_text", regexp_replace("cleaned_text", r"#\w+", ""))
    cleaned_df = cleaned_df.withColumn("cleaned_text", regexp_replace("cleaned_text", r"[^a-zA-Z\s]", ""))
    cleaned_df = cleaned_df.withColumn("cleaned_text", trim(regexp_replace("cleaned_text", r"\s+", " ")))

    # 3. Remove Stopwords
    cleaned_df = cleaned_df.withColumn("text_array", split(col("cleaned_text"), " "))
    remover = StopWordsRemover(inputCol="text_array", outputCol="filtered_words")
    cleaned_df = remover.transform(cleaned_df)
    cleaned_df = cleaned_df.withColumn("final_text", concat_ws(" ", "filtered_words"))

    # 4. Select final columns and save
    final_df = cleaned_df.select("sentiment", "final_text").filter(col("final_text") != "")
    
    final_df.coalesce(1).write.csv(output_path, header=True, mode="overwrite")
    print(f"Successfully processed and saved to {output_path}")

# --- Define Paths ---
# These paths are correct assuming you run from the project root.
train_input_path = os.path.join("data", "twitter_training.csv")
val_input_path = os.path.join("data", "twitter_validation.csv")

train_output_path = os.path.join("output", "processed_data", "train_processed")
val_output_path = os.path.join("output", "processed_data", "validation_processed")

# --- Run Processing ---
process_file(train_input_path, train_output_path)
process_file(val_input_path, val_output_path)

print("Preprocessing complete for both files.")
spark.stop()

