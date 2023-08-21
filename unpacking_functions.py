import re
import datetime
import pandas as pd

def games_owned(review):
    return int(re.split(',|\:', str(review))[3])



def games_reviewed(review):
    return int(re.split(',|\:', str(review))[5])



def total_playtime(review):
    return float(re.split(',|\:', str(review))[7])



def playtime_last_2_weeks(review):
    return int(re.split(',|\:', str(review))[9])



def playtime_at_review(review):
    return int(re.split(',|\:', str(review))[11])



def last_played(review):
    return datetime.datetime.fromtimestamp(int(re.split(',|\:|\}', str(review))[-2]))



def word_count(review):
    return len(str(review).split())


def review_type(data):
    if data:
        return 'Positive'
    if not data:
        return 'Negative'
    

def is_joke(data):
    if data >= (1/3):
        return 1
    else:
        return 0
    
    
def is_helpful(data):
    if data <= 0.1:
        return "Not Helpful"
    else:
        return "Helpful"