import tweepy
import csv
from textblob import TextBlob

consumer_key = 'uD0MWCErmZm53xShW8RDabQWD'
consumer_secret ='LDUeDJQHC8IBN2lGOl4EA7uD1Dy8eDM5G4FDxp6y7fLkMb2Ce7'

access_token ='705354283353374720-QODsU3a5ysJjT0ta2qIn6HJmfiInogk'
access_token_secret = 'n8XxfTDdOHt2rvM7Dqdi2Xcu9q43gN3bXFt1b1VpdFwYG'

auth= tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Trump')

for tweet in public_tweets:
    a = tweet.text
    analysis =TextBlob(tweet.text)
    b = analysis.sentiment
    c = [a, b]
    with open ('tweets.csv' , mode='a') as tweety:
               them_tweets= csv.writer( tweety, delimiter= ',', quotechar= "'", quoting= csv.QUOTE_MINIMAL )
               them_tweets.writerow([c])
