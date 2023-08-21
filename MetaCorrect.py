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
        
      
        return self
    
    def transform(self, X):
       
        return np.array(self.model.predict(X)).reshape(-1,1)


references = pd.read_csv('references.csv')    
    
st.title("MetaCorrect")

st.text("""This app uses a machine learning model to predict helpfulness scores for Steam 
reviews and produces a weighted metascore based on these predictions""")

app = st.text_input("Enter the name of Steam game here (if the name doesn't work, try the AppID):", value = "Tales of Arise")

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

metascore = function_deck.augmented_metascore(predictions)

chart = function_deck.make_chart(predictions)

st.subheader(metascore)

col1, col2 = st.columns(2)

with col1:
    st.subheader(":blue[Positive Reviews]")
    
    pos_helpful_percent = function_deck.pos_helpful_percentage(predictions)
    
    pos_unhelpful_percent = function_deck.pos_unhelpful_percentage(predictions)
    
    st.markdown("### Helpful: {percentage} \%".format(percentage = pos_helpful_percent))
    
    st.markdown("### Not Helpful: {percentage} \%".format(percentage = pos_unhelpful_percent))
    
with col2:
    st.subheader(":red[Negative Reviews]")
    
    neg_helpful_percent = function_deck.neg_helpful_percentage(predictions)

    neg_unhelpful_percent = function_deck.neg_unhelpful_percentage(predictions)
    
    st.markdown("### Helpful: {percentage} \%".format(percentage = neg_helpful_percent))
    
    st.markdown("### Not Helpful: {percentage} \%".format(percentage = neg_unhelpful_percent))
    
    
    

st.subheader(":blue[Top positive review]")
top_3_pos = function_deck.top_3_positive(predictions)
    
top_pos_text = top_3_pos[0].split()
top_pos_text_truncated = " ".join(top_pos_text[0:100])
    
st.write(top_pos_text_truncated + '...')
    
st.write('Truncated, see top reviews page for full text and additional reviews')
    
    
    


st.subheader(":red[Top negative review]")
top_3_neg = function_deck.top_3_negative(predictions)
    
top_neg_text = top_3_neg[0].split()
top_neg_text_truncated = " ".join(top_neg_text[0:100])
    
st.write(top_neg_text_truncated + '...')
    
st.write('Truncated, see top reviews page for full text and additional reviews')








