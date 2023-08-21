import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import re
import json
import requests
import unpacking_functions
import function_deck
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from spacy.lang.en.stop_words import STOP_WORDS
import time

class ModelTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        # Fit the stored predictor.
        # Question: what should be returned?
      
        return self
    
    def transform(self, X):
        # Use predict on the stored predictor as a "transformation".
        # Be sure to return a 2-D array.
        return np.array(self.model.predict(X)).reshape(-1,1)


references = pd.read_csv('references.csv')

app = st.text_input("Enter the name of Steam game here (if the name doesn't work, try the AppID):")

if app == '':
    time.sleep(3600)

if app.isnumeric():
    appid = app

else:
    try:
        appid = function_deck.name_to_id(app)
    except ValueError:
        e = ValueError('Game name is not in reference list. Please try the game\'s AppID instead.')
        raise st.exception(e)
    


reviews = function_deck.get_reviews(appid)

predictions = function_deck.generate_predictions(reviews)



st.header("Top reviews by category")

st.subheader(":blue[Top 3 positive reviews]")
top_3_pos = function_deck.top_3_positive(predictions)
    
st.write(top_3_pos[0])
    
st.text("------------")
    
st.write(top_3_pos[1])
    
st.text("------------")
    
st.write(top_3_pos[2])
    
st.subheader(":red[Top 3 negative reviews]")
top_3_neg = function_deck.top_3_negative(predictions)
    
st.write(top_3_neg[0])
    
st.text("------------")
    
st.write(top_3_neg[1])
    
st.text("------------")
    
st.write(top_3_neg[2])




