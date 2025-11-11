import os
import glob
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

print("Starting visualization...")

# --- Define Paths ---
# MODIFICATION: Removed ".." from the paths.
data_dir = os.path.join("output", "processed_data", "train_processed")
visuals_path = os.path.join("output", "visuals")
os.makedirs(visuals_path, exist_ok=True)

# Load data
try:
    # MODIFICATION: Made glob pattern specific to Spark's output
    search_path = os.path.join(data_dir, "part-*.csv")
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        raise IndexError # Trigger the error block below
        
    csv_file = csv_files[0]
    print(f"Found data file: {csv_file}")
    df = pd.read_csv(csv_file)
    
except IndexError:
    print(f"Error: No 'part-*.csv' file found in {data_dir}.")
    print("Did 'preprocessing.py' run successfully?")
    exit()

# Handle potential nulls and ensure sentiment is lowercase
df = df.dropna(subset=['final_text', 'sentiment'])
df['final_text'] = df['final_text'].astype(str)
df['sentiment'] = df['sentiment'].str.lower()

print(f"Loaded {len(df)} cleaned training tweets for visualization.")

# Function to generate and save a word cloud
def create_word_cloud(text_data, filename):
    if not text_data:
        print(f"No data to generate word cloud for {filename}")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    save_path = os.path.join(visuals_path, filename)
    plt.savefig(save_path)
    print(f"Saved {filename} to {visuals_path}")
    plt.close() # Close the plot to save memory

# 1. Separate text by sentiment
positive_text = " ".join(tweet for tweet in df[df['sentiment'] == 'positive']['final_text'])
negative_text = " ".join(tweet for tweet in df[df['sentiment'] == 'negative']['final_text'])

# 2. Generate Word Clouds
create_word_cloud(positive_text, "positive_wordcloud.png")
create_word_cloud(negative_text, "negative_wordcloud.png")

print("Word cloud generation complete.")

# 3. Plot Sentiment Distribution
print("Generating sentiment distribution plot...")
plt.figure(figsize=(7, 5))
# Order the sentiments for a consistent plot
sentiment_order = ['positive', 'neutral', 'negative']
sentiment_counts = df['sentiment'].value_counts().reindex(sentiment_order).fillna(0)

sentiment_counts.plot(kind='bar', color=['green', 'blue', 'red'])
plt.title('Sentiment Distribution of Training Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=0)
dist_path = os.path.join(visuals_path, "sentiment_distribution.png")
plt.savefig(dist_path)
plt.close()
print(f"Saved sentiment distribution plot to {dist_path}")

print("Visualization complete.")