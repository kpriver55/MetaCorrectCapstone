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

st.header('Extra Analysis')

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

chart = function_deck.make_chart(predictions)


st.write("This chart compares the number of helpful vs. unhelpful reviews in the sample divided by review type")

st.altair_chart(chart)



st.text("""
This chart shows the top 30 most positively polar phrases in the sample.
This means that these phrases are likely to appear in positive reviews and unlikely
to appear in negative reviews.
""")

polarity_chart_plus = function_deck.make_polarity_chart(function_deck.get_plus(function_deck.get_polarities(reviews)))
        
st.altair_chart(polarity_chart_plus)

st.text("""
The next chart is the same as the above, but for negatively polar phrases.
""")

polarity_chart_minus = function_deck.make_polarity_chart(function_deck.get_minus(function_deck.get_polarities(reviews)))

st.altair_chart(polarity_chart_minus)