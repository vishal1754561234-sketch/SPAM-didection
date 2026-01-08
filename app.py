import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the trained model, vectorizer, and label encoder
try:
    model = joblib.load('logistic_regression_sms_spam_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("Model, vectorizer, or label encoder files not found. Please ensure they are saved in the correct location.")
    st.stop()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_sms(sms_text):
    """
    Applies the defined preprocessing steps to a single SMS message.

    Args:
        sms_text (str): The input SMS message string.

    Returns:
        str: The processed text string, ready for vectorization.
    """
    # 1. Remove special characters and numbers
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', sms_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # 2. Tokenization
    tokens = nltk.word_tokenize(cleaned_text)

    # 3. Removing Stopwords
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # 4. Lemmatization
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]

    # Convert tokens back to a string for vectorization
    processed_text = ' '.join(tokens)

    return processed_text

def predict_sms(sms_text, model, vectorizer, label_encoder):
    """
    Predicts whether a new SMS message is spam or ham.

    Args:
        sms_text (str): The input SMS message string.
        model: The trained Logistic Regression model.
        vectorizer: The fitted TF-IDF vectorizer.
        label_encoder: The fitted Label Encoder.


    Returns:
        str: 'spam' or 'ham'.
    """
    # Preprocess the new SMS
    processed_sms = preprocess_sms(sms_text)
    if not processed_sms: # Handle empty string after preprocessing
        return "Cannot classify empty or highly filtered message."

    # Vectorize the processed SMS
    vectorized_sms = vectorizer.transform([processed_sms])

    # Predict the label
    prediction = model.predict(vectorized_sms)

    # Decode the predicted label back to 'ham' or 'spam'
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    return predicted_label

# Streamlit App
st.title("SMS Spam Detection")

st.write("Enter an SMS message below to check if it is Spam or Ham.")

sms_input = st.text_area("Enter SMS Message:", height=150)

if st.button("Predict"):
    if sms_input:
        prediction = predict_sms(sms_input, model, tfidf_vectorizer, le)
        if prediction == 'spam':
            st.error(f"Prediction: {prediction.upper()}")
        else:
            st.success(f"Prediction: {prediction.upper()}")
    else:
        st.warning("Please enter an SMS message to predict.")

