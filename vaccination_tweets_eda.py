# given tweets about the Pfizer vaccinations/COVID19 lets try predicting the sentiment of the tweets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

#loading the dataset
df = pd.read_csv('vaccination_tweets.csv', encoding='latin-1')
#information about the data set/ we can see that there are some missing values in unser location/description
df.info()

#drop the names column
df.drop(columns='user_name')
print(df)

print('length of data is', len(df))

# any null values
np.sum(df.isnull().any(axis=1))

# account verified or not
df['user_verified']=df['user_verified'].apply(lambda x:'verified' if x==True else 'not_verified')
print(df)

#total engagement
df['total_engagement']=df['retweets']+df['favorites']
print(df)

#location
df['user_location'].value_counts()
print(df)

#create a positive and negative sentiment column for each tweet
analyser = SentimentIntensityAnalyzer()
scores=[]
for i in range(len(df['text'])):
    
    score = analyser.polarity_scores(df['text'][i])
    score=score['compound']
    scores.append(score)
sentiment=[]
for i in scores:
    if i>=0.05:
        sentiment.append('Positive')
    elif i<=(-0.05):
        sentiment.append('Negative')
    else:
        sentiment.append('Neutral')
df['sentiment']=pd.Series(np.array(sentiment))

temp = df.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)
temp.style.background_gradient(cmap='Purples')

# creating a subset of location and then putting it into a bar graph to show where the most tweets have come from

plt.figure(figsize=(10,12))
sns.barplot(df['user_location'].value_counts().values[0:10],
            df['user_location'].value_counts().index[0:10])
plt.title('Top 10 Countries with Maximum Tweets', fontsize=14)
plt.xlabel('Number of Tweets', fontsize=14)
plt.ylabel('Country', fontsize=14)
plt.show()

# verified users plot to show the credibility
plt.figure(figsize=(5,5))
sns.countplot(x ="user_verified", data=df, palette='Set1')
plt.title("Verified User Accounts or Not?")
plt.xticks([False,True], ['Unverified', 'Verified'])
plt.show()

#plot correlation between features
plt.figure(figsize=(10,8))
sns.heatmap(df.drop(columns=['id', 'is_retweet']).corr(), square=True, annot=True)
plt.show()

# cleaning the tweets from punctuations
def clean_text(text):
    text = str(text).lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text
df['text'] = df['text'].apply(lambda x:clean_text(x))

# cleaning data of stop words/ tokenised df2

df2 =pd.DataFrame()
df2['text']=df['text']
def tokenization(text):
    text = re.split('\W+', text)
    return text
df2['tokenized'] = df2['text'].apply(lambda x: tokenization(x.lower()))
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
    
df2['No_stopwords'] = df2['tokenized'].apply(lambda x: remove_stopwords(x))
ps = nltk.PorterStemmer()
def stemming1(text):
    text = [ps.stem(word) for word in text]
    return text
df2['stemmed_porter'] = df2['No_stopwords'].apply(lambda x: stemming1(x))
from nltk.stem.snowball import SnowballStemmer
s_stemmer = SnowballStemmer(language='english')
def stemming2(text):
    text = [s_stemmer.stem(word) for word in text]
    return text
df2['stemmed_snowball'] = df2['No_stopwords'].apply(lambda x: stemming2(x))

#cleaning repeating characters
def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)
df2['text'] = df2['text'].apply(lambda x: cleaning_repeating_char(x))
df2['text'].tail()

#cleaning www and urls
def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)
df2['text'] = df2['text'].apply(lambda x: cleaning_URLs(x))
df2['text'].tail()

#cleaning and removing numbers
def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
df2['text'] = df2['text'].apply(lambda x: cleaning_numbers(x))
df2['text'].tail()

#lemmatise df2
wn = nltk.WordNetLemmatizer()
def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text
df2['lemmatized'] = df2['No_stopwords'].apply(lambda x: lemmatizer(x))
df2.head()

# add lemmatized text to main df
df['text'] = df2['lemmatized']

#define the data
x = df['text']
y = df['sentiment']

print(len(x), len(y))

#train/split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state= 34)

print(len(x_train), len(y_train))
print(len(x_test), len(y_test))
