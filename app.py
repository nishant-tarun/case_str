import os
import re
import glob
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# from openai import OpenAI  # <-- REMOVED
import google.generativeai as genai  # <-- ADDED

# --- Page Configuration ---
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- NLTK Stopwords (for live prediction cleaning) ---
@st.cache_data
def download_nltk_stopwords():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
download_nltk_stopwords()
stop_words = set(stopwords.words('english'))

# --- File Paths ---
MODEL_PATH = os.path.join("output", "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join("output", "models", "tfidf_vectorizer.pkl")
# Load the PROCESSED TRAINING data for insights
PROCESSED_DATA_DIR = os.path.join("output", "processed_data", "train_processed") # <-- UPDATED
VISUALS_DIR = os.path.join("output", "visuals")

# --- Caching: Load Models and Data ---
@st.cache_resource
def load_models_and_data():
    """Load the trained model, vectorizer, and cleaned data."""
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        
        # Find the CSV file written by PySpark in the 'train_processed' folder
        csv_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No processed CSV file found in {PROCESSED_DATA_DIR}")
        
        df = pd.read_csv(csv_files[0])
        # Clean data on load
        df = df.dropna(subset=['final_text', 'sentiment'])
        df['sentiment'] = df['sentiment'].str.lower()
        df = df[df['sentiment'] != 'irrelevant']
        return model, vectorizer, df
    except FileNotFoundError as e:
        print(f"Error loading assets: {e}")
        return None, None, None

# --- Text Cleaning Function (for live prediction) ---
def clean_text_for_prediction(text):
    """Cleans a single string for prediction."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# --- Google Gemini API Client ---
@st.cache_resource
def get_gemini_client():
    """Initialize Google Gemini client using Streamlit secrets."""
    # This requires the .streamlit/secrets.toml file
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        return model
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        return None
    
# --- Load all assets ---
model, vectorizer, df = load_models_and_data()
gemini_model = get_gemini_client()  # <-- CHANGED

# --- App UI ---
if model is None or df is None:
    st.error(
        "🚨 **Error: Model or Data files not found!**\n\n"
        f"Could not find assets in `{os.path.join('output', 'models')}` or `{PROCESSED_DATA_DIR}`.\n\n"
        "**Please run your offline pipeline scripts first (in order):**\n"
        "1. `python src/preprocessing.py`\n"
        "2. `python src/train_model.py`\n"
        "3. `python src/visualize.py`"
    )
else:
    st.sidebar.title("Sentiment Project Dashboard")
    page = st.sidebar.radio(
        "Navigate to:",
        (
            "Overview", 
            "Live Sentiment Analysis", 
            "Dataset Insights", 
            "Ad Slogan Generator", 
            "Privacy & Security Discussion"
        )
    )

    # --- Page 1: Overview ---
    if page == "Overview":
        st.title("📊 Social Media Brand Reputation Tracker")
        st.markdown(f"""
        This dashboard presents the results of the sentiment analysis pipeline.
        The data was processed using **PySpark** (`01_preprocessing.py`) and
        a **Logistic Regression** model was trained (`02_train_model.py`).
        
        The training dataset contains **{len(df)}** processed tweets.
        
        Use the sidebar to navigate to the different modules of this application.
        """)
        
        st.subheader("Training Data Insights")
        st.write("These are the visualizations from the training dataset, created by `src/visualize.py`.")
        try:
             st.image(os.path.join(VISUALS_DIR, "sentiment_distribution.png"), caption="Sentiment Distribution")
        except FileNotFoundError:
            st.warning("Could not find `sentiment_distribution.png`. Did `src/visualize.py` run successfully?")

    # --- Page 2: Live Sentiment Analysis ---
    elif page == "Live Sentiment Analysis":
        st.header("🔬 Live Sentiment Analyzer")
        st.write("Enter any text (e.g., a tweet or review) to see its predicted sentiment.")
        
        user_input = st.text_area("Your text:", "This product is absolutely fantastic! I love it.", height=100)
        
        if st.button("Analyze Sentiment"):
            if user_input:
                cleaned_input = clean_text_for_prediction(user_input)
                if not cleaned_input:
                    st.warning("Input text was empty after cleaning. Please enter more text.")
                else:
                    vectorized_input = vectorizer.transform([cleaned_input])
                    prediction = model.predict(vectorized_input)[0]
                    proba = model.predict_proba(vectorized_input)[0]
                    proba_percent = max(proba) * 100
                    
                    if prediction == 'positive':
                        st.success(f"**Positive** (Confidence: {proba_percent:.2f}%)", icon="😊")
                    elif prediction == 'negative':
                        st.error(f"**Negative** (Confidence: {proba_percent:.2f}%)", icon="😠")
                    else:
                        st.info(f"**Neutral** (Confidence: {proba_percent:.2f}%)", icon="😐")
                    
                    st.dataframe(pd.DataFrame([proba], columns=model.classes_))
            else:
                st.warning("Please enter some text to analyze.")

    # --- Page 3: Dataset Insights ---
    elif page == "Dataset Insights":
        st.header("📈 Training Dataset Insights")
        st.write(f"These visualizations are the **pre-computed** images from `{VISUALS_DIR}`.")
        
        st.subheader("Sentiment Distribution")
        try:
            st.image(os.path.join(VISUALS_DIR, "sentiment_distribution.png"),use_container_width=True)
        except:
            st.error("Could not load `sentiment_distribution.png`.")

        st.divider()
        st.subheader("Word Clouds")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Positive Tweets**")
            try:
                st.image(os.path.join(VISUALS_DIR, "positive_wordcloud.png"),use_container_width=True)
            except:
                st.error("Could not load `positive_wordcloud.png`.")
        with col2:
            st.write("**Negative Tweets**")
            try:
                st.image(os.path.join(VISUALS_DIR, "negative_wordcloud.png"),use_container_width=True)
            except:
                st.error("Could not load `negative_wordcloud.png`.")

    # --- Page 4: Ad Slogan Generator ---
    elif page == "Ad Slogan Generator":
        st.header("💡 AI Ad Slogan Generator")
        st.write("Generate creative ad slogans based on real positive customer feedback from the dataset.")
        
        if gemini_model is None:  # <-- CHANGED
            st.error(
                "🚨 **Google API Key not found.**\n\n"  # <-- CHANGED
                "To use this feature, please create a file at `.streamlit/secrets.toml` "
                "and add your key:\n"
                "`GOOGLE_API_KEY = 'your_key_here'`"  # <-- CHANGED
            )
        else:
            if st.button("Generate 5 Ad Slogans"):
                with st.spinner("🧠 AI is thinking..."):
                    try:
                        positive_tweets = df[df['sentiment'] == 'positive']['final_text'].sample(10).tolist()
                        prompt_text = f"Based on these positive customer reviews:\n{'- '.join(positive_tweets)}\n\nGenerate 5 short, catchy ad slogans:"
                        
                        # --- ENTIRE API CALL BLOCK CHANGED ---
                        response = gemini_model.generate_content(prompt_text)
                        slogans = response.text
                        # --- END OF CHANGE ---
                        
                        st.success("Here are your AI-generated slogans!")
                        st.markdown(slogans)
                        
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

    # --- Page 5: Privacy & Security Discussion ---
    elif page == "Privacy & Security Discussion":
        st.header("🔒 Privacy, Security, and Ethical Risks")
        st.markdown(
            """
            Using User-Generated Content (UGC) from social media is powerful, but it carries significant ethical and technical risks.
            
            ### 1. Data Privacy and Ethical Concerns
            * **Data Scraping vs. API Use:** Scraping data (like this dataset was) often violates platforms' Terms of Service (ToS).
            * **User Privacy and PII:** Posts can contain Personally Identifiable Information (PII). Our preprocessing (removing `@mentions`) is a first step, but not foolproof.
            * **Risk of Re-identification:** Even in an "anonymized" dataset, a user can often be re-identified by cross-referencing their post content (the "data mosaic" effect).
            * **Informed Consent:** Users did not consent for their public tweets to be used to train a commercial AI model. This is a major ethical grey area.

            ### 2. Model Bias and Fairness
            * **Demographic Bias:** The Twitter dataset is not representative of the general population.
            * **Linguistic Bias:** The model may incorrectly classify certain dialects or slang (e.g., AAVE) as "negative."
            * **Sarcasm:** Models are notoriously bad at detecting sarcasm (e.g., "Great, my flight is delayed again.").

            ### 3. Security Risks
            * **Data Poisoning:** A competitor could intentionally spam your brand with thousands of fake, bot-generated tweets to "poison" the training data and corrupt your analysis.
            * **Credential Security:** The API key is a powerful secret. If leaked (e.g., hardcoded and pushed to GitHub), an attacker could run up thousands of dollars in charges. **Solution:** We use Streamlit's `.streamlit/secrets.toml` to keep it secure.
            """
        )



