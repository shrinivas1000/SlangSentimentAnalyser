import os
import streamlit as st
import numpy as np
import emoji
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences #type:ignore
from tensorflow.keras.models import load_model #type:ignore
from tensorflow.keras.layers import Layer #type:ignore
from keras.utils import custom_object_scope
import tensorflow.keras.backend as K #type:ignore


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = K.sum(context, axis=1)
        return context


with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)


with custom_object_scope({'Attention': Attention}):
    model = load_model('my_model.keras', compile=False)

max_len = 20

def preprocess_text(text):
    text = text.lower()
    text = emoji.demojize(text, delimiters=(' ', ' '))
    text = ' '.join(text.split())
    return text

def predict_sentiment(text):
    processed = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred_probs = model.predict(padded)
    pred_idx = np.argmax(pred_probs, axis=1)[0]
    pred_label = le.classes_[pred_idx]
    confidence = pred_probs[0][pred_idx]
    return pred_label, confidence

st.title("Gen-Z Slang Sentiment Analyser")

user_input = st.text_area("Enter text to analyse:")

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        label, confidence = predict_sentiment(user_input)
        st.markdown(f"**Sentiment:** {label}  \n**Confidence:** {confidence:.2f}")
