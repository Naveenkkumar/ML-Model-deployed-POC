import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

with open('model_lstm.pkl', 'rb') as file:
    model_lstm = pickle.load(file)

def predict_category(text):
    text = text.lower()

    #tokenize
    seq = tokenizer.texts_to_sequences([text])

    #padding
    padded_seq = pad_sequences(seq, maxlen=maxlen)

    #prediction
    pred = model_lstm.predict(padded_seq)

    print('This was predicted as :', pred)

    pred_label_index = np.argmax(pred, axis=1)
    pred_label = label_encoder.inverse_transform(pred_label_index)[0]
    return pred_label

#Example
#text = 'The Indian cricket team won the world cup'
#print(predict_category(text))



st.title('News Category Prediction')
st.write('This is a simple web app to predict the category of news article')

user_input = st.text_area('Enter the news article')

if st.button('Predict'):
    prediction = predict_category(user_input)
    st.write('The news article is of category :', prediction)
else:
    st.write('Please enter the news article')
    