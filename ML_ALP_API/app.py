import os
import re
import nltk
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the Flask application
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the model and tokenizer
model = tf.keras.models.load_model('cyberbullying_modelv3.h5')
with open('tokenizercyberv3.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

# Define labels for the predictions
reverse_encoding_map = {
    0: 'not_cyberbullying',
    1: 'gender',    2: 'religion',
    3: 'other_cyberbullying',
    4: 'age',
    5: 'ethnicity'
}

# Initialize NLTK components
nltk.download('stopwords')
nltk.download('vader_lexicon')
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

# Slang dictionary
slang_dict = {
    "idk": "i don't know",
    "wtd": "what to do",
    "btw": "by the way",
    "omg": "oh my god",
    "ttyl": "talk to you later",
    "brb": "be right back",
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "rofl": "rolling on the floor laughing",
    "smh": "shaking my head",
    "tbh": "to be honest",
    "wtf": "what the fuck",
    "omw": "on my way",
    "fyi": "for your information",
    "imo": "in my opinion",
    "ikr": "i know, right",
    "nvm": "never mind",
    "ftw": "for the win",
    "thx": "thanks",
    "np": "no problem",
    "gtg": "got to go",
    "afk": "away from keyboard",
    "bff": "best friends forever",
    "wyd": "what are you doing",
    "hbu": "how about you",
    "ily": "i love you",
    "omfg": "oh my fucking god",
    "gg": "good game",
    "gr8": "great",
    "yolo": "you only live once",
    "tmi": "too much information",
    "stfu": "shut the fuck up",
    "btfo": "back the fuck off",
    "wtg": "way to go",
    "srsly": "seriously",
    "jk": "just kidding",
    "hbd": "happy birthday",
    "asap": "as soon as possible",
    "fml": "fuck my life",
    "ty": "thank you",
    "np": "no problem",
    "ikr": "i know, right",
    "idc": "i don't care",
    "ttfn": "ta ta for now",
    "dw": "don't worry",
    "rn": "right now",
    "rt" : "retweet",
}

# Preprocessing function
def preprocess_and_analyze_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'http\S+|www\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    words = tweet.split()
    words = [slang_dict.get(word, word) for word in words]
    tweet = ' '.join(words)
    tokens = word_tokenize(tweet)
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_tweet = ' '.join(tokens)
    sentiment = sia.polarity_scores(cleaned_tweet)
    if sentiment['compound'] <= -0.5:
        cleaned_tweet += " [bullying]"
    return cleaned_tweet

# Prediction function
def predict_texts(texts):
    logging.debug(f"Received texts for prediction: {texts}")
    
    # Preprocess texts
    preprocessed_texts = [preprocess_and_analyze_tweet(text) for text in texts]
    
    sequences = tokenizer.texts_to_sequences(preprocessed_texts)
    
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=100)
    
    padded_sequences = np.array(padded_sequences)
    predictions = model.predict(padded_sequences)
    predictions = np.round(predictions, 4)
    logging.debug(f"Prediction probabilities: {predictions*100}")
    
    predicted_classes = np.argmax(predictions, axis=1)
    logging.debug(f"Predicted classes: {predicted_classes}")
    
    predicted_labels = [reverse_encoding_map[pred] for pred in predicted_classes]
    logging.debug(f"Predicted labels: {predicted_labels}")
    
    return predicted_labels

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '').strip()
    logging.debug(f"Received request with text: {text}")

    if not text:
        logging.warning("No text provided in the request.")
        return jsonify({'error': 'No text provided'}), 400

    predicted_label = predict_texts([text])[0]
    logging.info(f"Predicted label for input '{text}': {predicted_label}")

    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
