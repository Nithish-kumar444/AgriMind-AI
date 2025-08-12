## AgriMind AI 🌱🌾
An AI-powered multilingual crop recommendation and farming FAQ assistant for Indian farmers

## Project Overview
AgriMind AI is an intelligent agricultural assistant designed to help farmers select the best crop to cultivate based on soil and weather parameters. It also serves as a multilingual FAQ chatbot that answers common farming-related questions using relevant agricultural documents. The system supports multiple Indian languages and provides recommendations with confidence scores alongside context-based answers sourced from official guides and policies.

## Features

# 1. Crop Recommendation
Uses a trained Random Forest machine learning model to predict the most suitable crop based on input features:

Nitrogen (N) level

Phosphorus (P) level

Potassium (K) level

Temperature (°C)

Humidity (%)

Soil pH

Rainfall (mm)

# 2. Multilingual FAQ Chatbot
Answers farmers' questions in their native language by:

Translating the query to English

Retrieving relevant agricultural document chunks using semantic search (FAISS + Sentence Transformers)

Generating a context-based answer with a state-of-the-art text generation model (FLAN-T5)

Translating the answer back to the farmer’s language

# 3. Supports multiple Indian languages:
English, Hindi, Bengali, Marathi, Gujarati, Punjabi, Tamil, Telugu, Kannada, Malayalam, Odia

# 4. Interactive web UI powered by Gradio for easy access.

## Dataset Used
# 1. Crop Recommendation Dataset: 
Crop_recommendation.csv (soil and weather data with crop labels)

# 2. Farming FAQ Dataset: 
Farming_FAQ_Assistant_Dataset.csv (questions and answers dataset to build the FAQ knowledge base)

# 3. Agricultural Documents: 
Sample guides and policies saved as .txt files in /content/docs folder.

## How It Works

# 1. Data Preprocessing & Model Training

Loads crop dataset, processes features, and trains multiple classification models.

Random Forest model selected based on performance and saved using pickle.

MinMaxScaler used for feature normalization.

# 2. FAQ Knowledge Base Construction

Agricultural documents are chunked into smaller pieces.

Sentence embeddings computed using SentenceTransformer.

FAISS index built for fast semantic search.

# 3. Multilingual Translation & Text Generation

User queries translated to English using a multilingual translation pipeline.

Semantic search performed to find relevant context chunks.

Context and question fed to a FLAN-T5 model for answer generation.

Answer translated back to user’s language.

# 4. User Interface

Gradio app with inputs for farmer’s question, language, and soil/weather parameters.

Displays crop recommendation, AI-generated answer, and source documents.

## Installation
pip install -q transformers sentence-transformers faiss-cpu gradio pandas scikit-learn sentencepiece matplotlib seaborn wordcloud

## Usage
# 1. Prepare datasets:
Place Crop_recommendation.csv and Farming_FAQ_Assistant_Dataset.csv in your working directory.

# 2. Train and save the crop recommendation model:
Run the script to train the Random Forest model and save model.pkl and minmaxscaler.pkl.

# 3. Prepare agricultural documents:
Add .txt files containing agricultural policies and crop guides into /content/docs.

# 4. Run the Gradio app:
Launch the interactive interface to ask questions and get crop recommendations.

## Functions
 1. recommend_crop(N,P,K,temp,hum,ph,rain): Returns crop recommendation and confidence score.

 2. answer_query_multilingual(user_text, user_lang_code): Returns AI-generated answer and source docs in user language.

 3. full_chat(user_q, language, N,P,K,temp,hum,ph,rain): Combines crop recommendation and multilingual Q&A for the interface.

## Languages Supported

| Language  | Code |
| --------- | ---- |
| English   | en   |
| Hindi     | hi   |
| Bengali   | bn   |
| Marathi   | mr   |
| Gujarati  | gu   |
| Punjabi   | pa   |
| Tamil     | ta   |
| Telugu    | te   |
| Kannada   | kn   |
| Malayalam | ml   |
| Odia      | or   |

## Model Details

# 1. Crop Recommendation: RandomForestClassifier trained on soil and weather parameters.

# 2. Text Embedding: all-MiniLM-L6-v2 from Sentence Transformers.

# 3. Semantic Search: FAISS index for fast similarity search over document chunks.

# 4. Text Generation: google/flan-t5-small for context-aware answer generation.

# 5. Translation: facebook/m2m100_418M multilingual translation pipeline.

## Folder Structure

/content
  ├─ Crop_recommendation.csv
  ├─ Farming_FAQ_Assistant_Dataset.csv
  ├─ model.pkl
  ├─ minmaxscaler.pkl
  ├─ /docs
      ├─ agriculture_act.txt
      ├─ rice_guide.txt
      └─ wheat_guide.txt
  ├─ Ai_Agent.ipynb

## Acknowledgements

1. scikit-learn for ML algorithms

2. Sentence Transformers for semantic embeddings

3. FAISS for similarity search

4. Huggingface Transformers for translation and generation models

5. Gradio for web interface

6. Datasets and agricultural document sources

License
This project is for educational and research purposes. Please attribute appropriately.
