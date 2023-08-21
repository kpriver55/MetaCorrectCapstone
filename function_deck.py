import datetime
import re
import matplotlib.pyplot as plt
import altair as alt
import json
import requests
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from spacy.lang.en.stop_words import STOP_WORDS
import unpacking_functions


class ModelTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        
      
        return self
    
    def transform(self, X):
        
        return np.array(self.model.predict(X)).reshape(-1,1)
    




########    

def get_reviews(app):
    parameters = {'filter':'recent', 'langauge':'en',  'cursor' : '*', 
                      'review_type' : 'all', 'purchase_type' : 'all', 'num_per_page' : 100}
    for i in range(20):
        temp_url = "https://store.steampowered.com/appreviews/{appid}?json=1".format(appid = app)
        response = requests.get(temp_url, params = parameters)
        json = response.json()
        temp_df = pd.DataFrame.from_records(json['reviews'])
        if i == 0:
            reviews = temp_df
        else:
            reviews = pd.concat([reviews, temp_df])
        if len(json['reviews']) < 100:
            break
        
        parameters['cursor'] = json['cursor']
    reviews.dropna(axis = 0, subset = ['review'],inplace = True)
    reviews['games_owned'] = reviews['author'].apply(unpacking_functions.games_owned)
    reviews['games_reviewed'] = reviews['author'].apply(unpacking_functions.games_reviewed)
    reviews['total_playtime'] = reviews['author'].apply(unpacking_functions.total_playtime)
    reviews['playtime_last_2_weeks'] = reviews['author'].apply(unpacking_functions.playtime_last_2_weeks)
    reviews['playtime_at_review'] = reviews['author'].apply(unpacking_functions.playtime_at_review)
    reviews['last_played'] = reviews['author'].apply(unpacking_functions.last_played)
    reviews['word_count'] = reviews['review'].apply(unpacking_functions.word_count)
    reviews['joke_rating'] = reviews['votes_funny']/reviews['votes_up']
    reviews['joke'] = reviews['joke_rating'].apply(unpacking_functions.is_joke)
    reviews['review_type'] = reviews['voted_up'].apply(unpacking_functions.review_type)
    
    return reviews


#######

def generate_predictions(data):
    
    inputs = data.drop(['weighted_vote_score', 'joke'], axis = 1) 
    augmented_metascore_model = joblib.load('augmented_metascore_model.sav')
    joke_detector = joblib.load('joke_detector.sav')
    predictions = augmented_metascore_model.predict(inputs)
    joke_predictions = joke_detector.predict(inputs)
    predictions_normalized = (predictions - np.min(predictions))/(np.max(predictions) - np.min(predictions))
    
    predictions = pd.Series(predictions_normalized)
    #data = {'helpfulness_score' : predictions, 'review_type': data['review_type'], 'review_text' : data['review'] }
    #pred_frame = pd.DataFrame(data)
    jokes = pd.Series(joke_predictions).tolist()
    scores = pd.Series(predictions).tolist()
    word_counts = pd.Series(data['word_count']).tolist()
    for i in range(len(jokes)):
        if jokes[i] == 1 or word_counts[i] < 100:
            scores[i] /= 10
        
    scores = pd.Series(scores)
    
    
    pred_frame = scores.to_frame(name = 'helpfulness_score')
    
    pred_frame['joke'] = pd.Series(jokes)
    pred_frame['review_type'] = pd.Series(data['review_type'].tolist())
    pred_frame['review_text'] = pd.Series(data['review'].tolist())
    
    
    return pred_frame

######


def augmented_metascore(data):
    votes = data['review_type']
    scores = data['helpfulness_score']
    positives = 0
    score = 0
    for i in range(len(votes)):   
        if votes[i] == "Positive":
            score += scores[i]
            positives += 1
    
    score /= scores.sum()
    
    raw_score = positives/len(votes)
    return 'Augmented Score:' + ' ' + str(round(100*score, 2)) + ' ' +  '|' + ' ' 'Raw Score:' + ' ' + str(round(100*raw_score))

######

def make_chart(data):
    data['Helpful?'] = data['helpfulness_score'].apply(unpacking_functions.is_helpful)
    return alt.Chart(data, title = "Helpful vs. Unhelpful Review Totals").mark_bar().encode(alt.X("Helpful?"), 
                                                          y = alt.Y('count()').stack(None).axis().title('Number of Reviews'), 
                                                          color = alt.Color('review_type').scale(scheme = 'redblue'), 
                                                          column = 'review_type'
                                                          ).properties(width = 200).configure_title(anchor = 'middle')


######

def name_to_id(game):
    references = pd.read_csv('references.csv')
    game_list = references['name'].tolist()
    index = game_list.index(game)
    return str(references['appid'][index])


######
def get_polarities(reviews):
    review_text = reviews[['review', 'review_type']]
    Stop_Words = STOP_WORDS.union({'ve', 'good', 'best', 'game', 'games', 'll', 'ass', 'feels' })
    tfidf = TfidfVectorizer(min_df = 0.003, ngram_range = (2,3), stop_words = list(Stop_Words))
    counts = tfidf.fit_transform(review_text['review'])
    nb = MultinomialNB()
    nb.fit(counts, reviews['review_type'])
    words = tfidf.get_feature_names_out()
    words = pd.Series(words).to_frame(name = 'word')
    log_probs = pd.DataFrame(nb.feature_log_prob_)
    polarity_score_table = (log_probs.iloc[1] - log_probs.iloc[0]).to_frame(name = 'polarity_score') 
    polarity_score_table = pd.concat([words, polarity_score_table], axis = 1)
    polarity_score_table = polarity_score_table.sort_values('polarity_score')
    top_30_plus_minus = pd.concat([polarity_score_table.head(30), polarity_score_table.tail(30)])
    return top_30_plus_minus


######

def get_minus(polarities):
    return polarities.head(30)


######

def get_plus(polarities):
    return polarities.tail(30)

######

def make_polarity_chart(data):
    return alt.Chart(data).mark_bar().encode(x = 'polarity_score:Q', y = alt.X('word').sort('-x'),
                                             color = alt.condition(alt.datum.polarity_score > 0, 
                                                                   alt.value('steelblue'),
                                                                   alt.value('red')
                                                                  )).properties(height = 600)

######

def top_3_positive(data):
    pos_data = data[data['review_type'] == 'Positive']
    pos_data = pos_data.sort_values('helpfulness_score')
    pos_data['word_count'] = pos_data['review_text'].apply(unpacking_functions.word_count)
    top_3 = pos_data.tail(3).sort_values('word_count', ascending = False)
    
    
    return top_3['review_text'].tolist()


######

def top_3_negative(data):
    neg_data = data[data['review_type'] == 'Negative']
    neg_data = neg_data.sort_values('helpfulness_score')
    neg_data['word_count'] = neg_data['review_text'].apply(unpacking_functions.word_count)
    top_3 = neg_data.tail(3).sort_values('word_count', ascending = False)
    
    
    return top_3['review_text'].tolist()


######

def neg_helpful_percentage(data):
    data['Helpful?'] = data['helpfulness_score'].apply(unpacking_functions.is_helpful)
    neg_data = data[data['review_type'] == 'Negative']
    neg_helpful_data = neg_data[neg_data['Helpful?'] == "Helpful"]
    neg_reviews = len(neg_data['review_type'].tolist())
    helpful_neg_reviews = len(neg_helpful_data['review_type'].tolist())
    percent = round(100*helpful_neg_reviews/neg_reviews)
    return percent

######

def neg_unhelpful_percentage(data):
    data['Helpful?'] = data['helpfulness_score'].apply(unpacking_functions.is_helpful)
    neg_data = data[data['review_type'] == 'Negative']
    neg_unhelpful_data = neg_data[neg_data['Helpful?'] == "Not Helpful"]
    neg_reviews = len(neg_data['review_type'].tolist())
    unhelpful_neg_reviews = len(neg_unhelpful_data['review_type'].tolist())
    percent = round(100*unhelpful_neg_reviews/neg_reviews)
    return percent

######

def pos_helpful_percentage(data):
    data['Helpful?'] = data['helpfulness_score'].apply(unpacking_functions.is_helpful)
    pos_data = data[data['review_type'] == 'Positive']
    pos_helpful_data = pos_data[pos_data['Helpful?'] == "Helpful"]
    pos_reviews = len(pos_data['review_type'].tolist())
    helpful_pos_reviews = len(pos_helpful_data['review_type'].tolist())
    percent = round(100*helpful_pos_reviews/pos_reviews)
    return percent

######

def pos_unhelpful_percentage(data):
    data['Helpful?'] = data['helpfulness_score'].apply(unpacking_functions.is_helpful)
    pos_data = data[data['review_type'] == 'Positive']
    pos_unhelpful_data = pos_data[pos_data['Helpful?'] == "Not Helpful"]
    pos_reviews = len(pos_data['review_type'].tolist())
    unhelpful_pos_reviews = len(pos_unhelpful_data['review_type'].tolist())
    percent = round(100*unhelpful_pos_reviews/pos_reviews)
    return percent






