import google.generativeai as genai
import os

# --- PASTE YOUR GEMINI API KEY HERE ---
# Make sure to put it inside the quotation marks
GOOGLE_API_KEY = "AIzaSyCMizFsh0KCmGBQMAbZ7LOI6COSZCTbSeM" 
# --------------------------------------

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error configuring API key: {e}")
    print("Please make sure your key is correct.")
    exit()

print("--- Finding all models available to your API key ---")
print("-" * 50)

found_model = False
for model in genai.list_models():
    # The error message says to check for 'generateContent' support
    if 'generateContent' in model.supported_generation_methods:
        print(f"✅ Found usable model: {model.name}")
        print(f"   (Display Name: {model.display_name})")
        print("-" * 50)
        found_model = True

if not found_model:
    print("❌ No models were found that support 'generateContent' for your API key.")
    print("This might be a permissions issue in your Google Cloud project.")

print("\n--- ACTION ---")
print("Copy one of the 'usable model' names (like 'models/gemini-1.5-flash-001')")
print("and paste it into your app.py file.")