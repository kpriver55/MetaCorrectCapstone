Documentation for MetaCorrect (WIP):

Summary:
This app uses a machine learning model to approximate the usefulness of user reviews for video games on Steam. 
It will provide an adjusted average review score using these usefulness scores as weights in the average as opposed
to weighting all reviews equally.

Usage:
To start up the app, navigate to the directory containing the files in this repo through the terminal.
Run the command "streamlit run MetaCorrect.py"
This will open the app on localhost. 
Warning: One should not run the app on more than 10 inputs every five minutes. This will cause the user to run afoul
of the Steam Store API's rate limit.

Input/Output:
The app takes the name of a game or a valid Steam AppID number as input. 
It will perform ingestion of 2000 user reviews and associated data for the inputted app in real time and return the following:
1. Raw Score: Percentage of reviews from the sample that are positive.
2. Augmented Score: Average positivity score based upon weights calculated by the underlying machine learning model.
3. Percentage of user reviews in each category that the model finds "helpful". We define helpful in this instance to be
reviews scoring above 0.1 by the underlying model. Note that scores fall between 0 and 1.
4. Excerpts from the top positive and top negative review, where "top" is defined by the model's scoring. 

Extra Pages:
1. Top Reviews: Outputs the full text of the top 3 reviews in each category
2. Extra Analysis: Provides additional insights through charts including a visual representation of the information in #3 of input/output and a charts of the top 30 phrases in positive and negative reviews as determined by "polarity score" (more info on technical details later).
3. About the Developer: Contact info

