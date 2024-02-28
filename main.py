import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

model_fraud = pickle.load(open('model_fraud.sav', 'rb'))

tfidf = TfidfVectorizer

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))))
st.set_page_config(
    page_title="SMS FRAUD DETECTOR",
    page_icon=":📲:",
)
import requests
from io import BytesIO

from PIL import Image
instagram_logo_url = 'https://www.freeiconspng.com/uploads/message-icon-png-3.png'
response = requests.get(instagram_logo_url)
if response.status_code == 200:
    instagram_logo = Image.open(BytesIO(response.content))

    # Resize the image to a smaller size
    smaller_logo = instagram_logo.resize((100, 100))  # Adjust the size as needed

    # Display the resized logo
    st.image(smaller_logo)
else:
    st.write("Gagal mengunduh logo Instagram")

def main():
    st.title("SMS SPAM Detector")

    message = st.text_area("Masukan")

    def detect_sentiment(input_text):
        predict_fraud = model_fraud.predict(loaded_vec.fit_transform([input_text]))

        if predict_fraud == 0:
            return 'SMS NORMAL'
        elif predict_fraud == 1:
            return 'SMS FRAUD'
        else:
            return 'SMS PROMO'

    if st.button("Deteksi SMS"):
        result = detect_sentiment(message)
        st.success(result)

if __name__ == "__main__":
    main()
