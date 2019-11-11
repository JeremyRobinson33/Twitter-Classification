#!/usr/bin/python3
# Jeremy Robinson
# November 2019
# Data Mining Twitter - Twitter Classification
# Using a Naive Bayes Classifier
# Required: pip install twitter, re, json, pandas, numpy, time, textblob
# Sklearn, maplotlib, seaborn
# Go to app.twitter.com to create your own app and generate keys

import re
import json
import twitter
import pandas as pd
import numpy as np
import time

from authTwitter import authTW
from textblob import TextBlob

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# clean the tweet of unwanted characters
def cleanTweet(t):
    # use the regular expression library to strip all unwanted characters from the text
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", t).split())

# determine if the tweet is positive, neutral
# or negative


def getSentiment(t):
    # using TextBlob, create an Object from the input tweet
    tbObject = TextBlob(t)

    # compute the sentiment
    if tbObject.sentiment.polarity > 0:
        return ('positive', 1)
    elif tbObject.sentiment.polarity == 0:
        return ('neutral', 0)
    else:
        return ('negative', -1)

# search tweets based off of a particular hashtag
# and return a list of tweets


def getSearch(t_obj, hashtag):
    q = '#' + hashtag
    count = 100

    # use the twitter api to get the tweets
    search_results = t_obj.search.tweets(q=q, locale="en", count=count)

    # filter the json results just to status
    statuses = search_results['statuses']

    # iterate through the status to get the tweet text and id
    # and calculate the sentiment of the tweet
    tw_list = []
    for tw in statuses:
        tw_text = tw['text']
        tw_text = cleanTweet(tw_text)

        tw_id = tw['id']
        sentWord, sentNum = getSentiment(tw_text)
        tw_list.append([tw_id, sentWord, tw_text, sentNum])

        # use TextBlob to analyze and compute sentiment for the text
        #print("\n ", getSentiment(tw_text), " : ", tw_text)

    return tw_list

# create a pandas dataframe from a list of tweets
# and then create a csv file from said dataframe


def toCSV(tweetList, csvName):
    # create the dataframe
    df = pd.DataFrame(tweetList)

    # set meaningful column names
    df = df.rename(columns={0: "Tweet ID", 1: "Sentiment",
                            2: "Tweet Text", 3: "Tweet Label"})

    # print(df)

    # create the csv file
    df.to_csv(csvName)
    return df

# get three hashtags from the user


def getHashtags():
    return list(map(str, input("Enter three hashtags ").strip().split()))

# create a list of tweets based on hashtags
# decided by the user


def constructTweeList(hashtagList, t_obj):
    tweetList = []
    for x in hashtagList:
        r = getSearch(t_obj, x)
        tweetList += r
    return tweetList


def main():
    # Used on first run to get initial set
    # of tweets
    # twitter_obj = authTW()
    # hashtagList = getHashtags()
    # tweetList = constructTweeList(hashtagList, twitter_obj)
    # df = toCSV(tweetList, 'twitter_sentiment.csv')

    # Read in the initial set
    df = pd.read_csv('twitter_sentiment2.csv')

    print("")
    print("Hastags")
    print("1. #FridayFeeling")
    print("2. #TrumpBribed")
    print("3. #PokemonSwordShield")
    print("")

    # create a NaiveBayes classifier instance
    clf = GaussianNB()
    naive_bayes = MultinomialNB()

    # divide the data into testing and training sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['Tweet Text'], df['Tweet Label'], test_size=0.33, random_state=15)

    print("Training Set")
    print(X_train.head())
    print("")

    print("Training the Naive Bayes Model...")

    # create a tfidf vectorizer so that the bayes
    # classifier can work with the tweet text
    tv = TfidfVectorizer()
    X_train_cv = tv.fit_transform(X_train)
    X_test_cv = tv.transform(X_test)
    # begin time to train classifier
    start_time = time.time()
    # fix classifier with the training data
    clf.fit(X_train_cv.toarray(), y_train)

    # calculate and print training time
    print("Training Time = ", (time.time()-start_time), "seconds")
    print("")

    predictions = clf.predict(X_test_cv.toarray())

    naive_bayes.fit(X_train_cv, y_train)
    Npredictions = naive_bayes.predict(X_test_cv)

    print("Test Set")
    print(X_test.head())
    print("")

    print("Running Test Set")

    # print out accuracy and precision scores
    print("==============================================================================")
    print('Accuracy score Gaussian: ', accuracy_score(y_test, predictions))
    print('Precision score Gaussian: ', precision_score(
        y_test, predictions, average='weighted'))

    print("==============================================================================")
    print('Accuracy score Multinomial: ', accuracy_score(y_test, Npredictions))
    print('Precision score Multinomial: ', precision_score(
        y_test, Npredictions, average='weighted'))
    print("")

    # get user input hashtags for generalization
    twitter_obj = authTW()
    hashtagList = getHashtags()
    tweetList = constructTweeList(hashtagList, twitter_obj)
    df2 = toCSV(tweetList, 'generalized.csv')

    print("")
    print("Running Generalized Test...")

    # create new feature and label vectors
    new_X = df2['Tweet Text']
    new_y = df2['Tweet Label']

    # transform to tfidf vector so
    # the classifier can understand the data
    new_X_predict = tv.transform(new_X)

    new_predictions = clf.predict(new_X_predict.toarray())

    new_Npredictions = naive_bayes.predict(new_X_predict)

    print("Generalized HashTags")
    for x in hashtagList:
        print("#"+x)

    print("")
    print("Generalized Set")
    print(new_X.head())
    print("")

    print("==============================================================================")
    print('Accuracy score Gaussian: ', accuracy_score(new_y, new_predictions))
    print('Precision score Gaussian: ', precision_score(
        new_y, new_predictions, average='weighted'))

    print("==============================================================================")
    print('Accuracy score Multinomial: ',
          accuracy_score(new_y, new_Npredictions))
    print('Precision score Multinomial: ', precision_score(
        new_y, new_Npredictions, average='weighted'))
    print("")

    # # create a heatmap of the initial test set
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, cmap='RdYlBu', annot=True, xticklabels=[
        'positive', 'neutral', 'negative'], yticklabels=['positive', 'neutral', 'negative'])
    # hm.figure.subplots_adjust(bottom=0.0)
    plt.margins(0.0)
    plt.title('Initial Test Set')
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()


main()
