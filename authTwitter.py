#!/usr/bin/python3
# Jeremy Robinson
# November 2019
# Data Mining Twitter - Twitter Auth example
# Required: pip install twitter
# Go to app.twitter.com to create your own app and generate keys

import twitter
from apiKeys import *


def authTW():
    auth = twitter.oauth.OAuth(
        OAUTH_TOKEN, OAUTH_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

    twitter_api = twitter.Twitter(auth=auth)

    return twitter_api


def authTWStream():
    auth = twitter.oauth.OAuth(
        OAUTH_TOKEN, OAUTH_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
    twitter_stream_api = twitter.TwitterStream(auth=auth)

    return twitter_stream_api


tw_obj = authTW()
print(tw_obj)
