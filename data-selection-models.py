'''

This code requires the following files to be present in the directory:

1. full set of tweets: "./twitter-news.csv"

2. hand-labelled tweets x4:

"./annotated_news_nl.csv"
"./annotated_news_it.csv"

"./annotated_replies_nl.csv"
"./annotated_replies_it.csv"


The packages and versions used are as follows (perhaps not exhaustive):


numpy==1.19.5
pandas==1.1.5
regex==2020.11.13
scikit-learn==0.23.2
sklearn==0.0
matplotlib==3.3.3
seaborn==0.11.2

huggingface-hub==0.4.0
sentencepiece==0.1.91
tensorflow==2.6.2     
tensorflow-estimator==2.6.0
tensorflow-hub==0.12.0

transformers==4.18.0  model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizers==0.12.1
bert-score==0.3.11

nltk==3.5
translatte==0.1
vader-multi==3.2.2.1  with bug fix available on: https://github.com/melaniewaterham/vader-multi


'''


# Read in big .csv file
import csv

'''Reads in a .csv file and outputs a generator with only the rows that match the language indicated in the column with index 1'''
def get_rows(crit, filename):
    with open("./twitter-news.csv") as file:
        for row in csv.reader(file):
            if row[1] == crit:
                yield row

with open("./tweets_NL.csv", 'w', newline='') as outfile:
    csv_writer = csv.writer(outfile)
    for row in get_rows('NL', "./twitter-news.csv"): # write NL tweets to new file
        csv_writer.writerow(row)
        
with open("./tweets_IT.csv", 'w', newline='') as outfile:
    csv_writer = csv.writer(outfile)
    for row in get_rows('IT', "./twitter-news.csv"): # write IT tweets to new file
        csv_writer.writerow(row)

        
# Read in the data as one df per country
import pandas as pd

df_nl = pd.read_csv("./tweets_NL.csv", header=None)
df_it = pd.read_csv("./tweets_IT.csv", header=None)

print(f' NL: {df_nl.shape}, IT: {df_it.shape}')


# Filter tweets by search terms
terms_nl = ['migratie', 'migrant', 'vluchteling', 'statushouder', 'veiligelander', 'asiel', 'Ter Apel']
terms_it = ['migrazione', 'migrant', 'immigrat' 'profug', 'rifugiat', 'clandestini', 'asilo', 'sbarco', 'sbarchi', 'Geo Barents', 'Humanity', 'Ocean Viking']

'''Keeps only the rows with tweets from a dataframe when the original tweet contains one of the search terms defined above'''
def filter_by_term(df, terms):
    mig_tweets = []
    for _, row in df.iterrows():
        if any(x in row[2] for x in terms):
            mig_tweets.append(row)
    mig_df = pd.DataFrame(mig_tweets).reset_index(drop=True)
    
    return mig_df

mig_nl = filter_by_term(df_nl, terms_nl) # Takes about 195 seconds for both dfs
mig_it = filter_by_term(df_it, terms_it)


# Drop lines that contain unrelated topics    
dump_nl = ['Qatar', 'Chile', 'aspergeteler', 'Capitoolrellen', 'beet', 'Icke']
dump_it = ['Maestra', 'maestra', 'nido', 'battaglione', 'Prokhorova', 'Mariupol'] # Missed two here: 'Vax' and 'Kiev'

'''Removes the rows with tweets (and their replies) that contain common noise topics using the search terms defined above'''
def dump_by_term(df, dump_terms):
    for _, row in df.iterrows():
        if any(x in row[2] for x in dump_terms):
            df.drop(_, axis=0, inplace=True)
    return df

mig_nl = dump_by_term(mig_nl, dump_nl)
mig_it = dump_by_term(mig_it, dump_it)


# Patterns for some initial tweet cleaning with regex
import re

OPTIONAL_LEFT = "?"
ANYTHING = "."
ZERO_OR_MORE = "*"
ONE_OR_MORE = "+"

START_OF_LINE = r"^"
NEWLINES = r"[\r\n]" # \r is a carriage return symbol, not sure if used in raw text of tweets but \n is
HASH = "#"
HASHTAGS = (HASH + "\w+")
MENTIONS = "@([a-zA-Z0-9_]{1,20})" # Twitter usernames have max 20 chars

# Patterns to remove links and text in square brackets/round brackets because it's usually the reporter's name
LINKS = ("http" + "s" + OPTIONAL_LEFT + ":" + "\/" "\/" + "\S" + ONE_OR_MORE)
TEXT_IN_BRACKETS = r"\[.*?\]"
NAME_IN_BRACKETS = r"\(di.*?\)" # deletes tags like: (di Giovanni Giorgio)


'''Selectively removes hyperlinks, the # symbol, tags in square brackets, and mentions from columns 2 and 4 containing the tweets to prepare for tokenization. Does not convert text to .lower() because VADER also uses capitalisation in sentiment scores, and it looks like RoBERTa uses it too.'''
def clean_tweets(df):
    df[2] = df[2].apply(lambda x: re.sub(LINKS, "", x))
    df[2] = df[2].apply(lambda x: re.sub(TEXT_IN_BRACKETS, "", x))
    df[2] = df[2].apply(lambda x: re.sub(NAME_IN_BRACKETS, "", x))
    df[2] = df[2].apply(lambda x: re.sub(HASH, "", x))
    # df[2] = df[2].apply(lambda x : x.lower())
    df[2] = df[2].apply(lambda x: re.sub(NEWLINES, " ", x))
    df[2] = df[2].apply(lambda x: re.sub(MENTIONS, "", x))

    df[4] = df[4].apply(lambda x: re.sub(LINKS, "", x))
    df[4] = df[4].apply(lambda x: re.sub(HASH, "", x))
    # df[4] = df[4].apply(lambda x : x.lower())
    df[4] = df[4].apply(lambda x: re.sub(NEWLINES, " ", x))
    df[4] = df[4].apply(lambda x: re.sub(MENTIONS, "", x))

    return df

clean_nl = clean_tweets(mig_nl)
clean_it = clean_tweets(mig_it)

print(f' replies NL all cleaned up: {mig_nl.shape}, replies IT all cleaned up: {mig_it.shape}')


# Splitting off the news tweet dfs and printing the number of on-topic news tweets per country
news_nl = clean_nl.drop_duplicates(subset=[2]).reset_index(drop=True)
news_it = clean_it.drop_duplicates(subset=[2]).reset_index(drop=True)                                 
                                   
print(f' clean news tweets NL: {news_nl.shape}, and clean news tweets IT: {news_it.shape}')


# ~~ VADER and RoBERTa sentiment scoring ~~


# Installing the multi-language VADER package and initialising classifier
import sys
!{sys.executable} -m pip uninstall vaderSentiment # Uninstall the original version so it is not instantiated instead of vader-multi:
!{sys.executable} -m pip install -e git+https://github.com/melaniewaterham/vader-multi.git#egg=vader-multi

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

'''Computes the VADER sentiment scores on the column with tweets that need to be scored, and creates additional columns to append info to the dataframe. Takes about 52 minutes: 39 min for the Dutch replies, 11 min for the Italian replies, and 1min40 for both news sets.'''
def add_score(df, tweet_col): # so I can choose the column with tweets by index
    df[5] = df[tweet_col].apply(lambda x: analyser.polarity_scores(x))
    df = pd.concat([df.drop([5], axis=1), df[5].apply(pd.Series)], axis=1)
    
    df['label'] = ['positive' if x >= 0.05 else 'negative' if x <= -0.05 else 'neutral' for x in df['compound']] # VADER rules
    return df

replies_nl = add_score(clean_nl, 4)
replies_it = add_score(clean_it, 4)

news_nl = add_score(news_nl, 2)
news_it = add_score(news_it, 2)


# Installing the transformers library and initialising model pipeline
import sys
!{sys.executable} -m pip install transformers

from transformers import pipeline
model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

'''Computes the RoBERTa label and probability score on the column with tweets that need to be scored, and creates 2 additional columns to display them in the dataframe'''
def add_more(df, tweet_col):
    #            | dict to columns | the column |             transformers     | 0th to get rid of list | 
    new_cols = pd.json_normalize(df[tweet_col].apply(lambda x: sentiment_task(x)[0]))
    # concatenate new_cols to old df 
    df = pd.concat([df, new_cols], axis=1)
    return df

replies_nl = add_more(replies_nl, 4)
replies_it = add_more(replies_it, 4)

news_nl = add_more(news_nl, 2)
news_it = add_more(news_it, 2)


'''Makes the column labels a bit clearer - renaming the RoBERTa labels'''
def rename_columns(df):
    df.columns = ['Outlet', 'Language', 'Tweet', 'Date', 'Reply', 'neg', 'neu', 'pos', 'VADER_compound_score', 'VADER_label', 'RoBERTa_label', 'RoBERTa_score']
    return df

rename_columns(replies_nl)
rename_columns(replies_it)

rename_columns(news_nl)
rename_columns(news_it)


# ~~~ Exploring the distribution of scores with plots for Q.2 ~~~

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2, style='white', palette='tab10')  # font_scale gives you big labels for manuscript

# Defining subsets of interest
total = pd.concat([replies_nl, replies_it, news_nl, news_it], ignore_index=True)

it = pd.concat([replies_it, news_it], ignore_index=True)
nl = pd.concat([replies_nl, news_nl], ignore_index=True)
news = pd.concat([news_nl, news_it], ignore_index=True)
replies = pd.concat([replies_nl, replies_it], ignore_index=True)


# Distribution of VADER scores by country (news)
fig, ax = plt.subplots(1, 2, figsize=(12,4), sharey=True)
plt.subplots_adjust(wspace=0.05) 

fig.suptitle('Distribution of VADER compound scores in the news tweets', fontsize=16, y=1.03)
ax[0].set_title('Language = NL')
ax[1].set_title('Language = IT')

sns.kdeplot(data=news_nl, x='VADER_compound_score', ax=ax[0], bw_adjust=.5)
sns.kdeplot(data=news_it, x='VADER_compound_score', ax=ax[1], bw_adjust=.5)


# Distribution of VADER scores by country (replies)
fig, ax = plt.subplots(1, 2, figsize=(12,4), sharey=True)
plt.subplots_adjust(wspace=0.05) 

fig.suptitle('Distribution of VADER compound scores in the reply tweets', fontsize=16, y=1.03)
ax[0].set_title('Language = NL')
ax[1].set_title('Language = IT')

sns.kdeplot(data=replies_nl, x='VADER_compound_score', ax=ax[0], bw_adjust=.5)
sns.kdeplot(data=replies_it, x='VADER_compound_score', ax=ax[1], bw_adjust=.5)


# RoBERTa scores by country with FacetGrid

g = sns.displot(data=news, x='RoBERTa_score', hue='RoBERTa_label', col='Language', kind='kde', bw_adjust=.75)        
g.fig.suptitle('Distribution of RoBERTa polarity scores in the news tweets', fontsize=16, y=1.05)
sns.move_legend(g, "center", bbox_to_anchor=(.55, .45))

g = sns.displot(data=replies, x='RoBERTa_score', hue='RoBERTa_label', col='Language', kind='kde', bw_adjust=.75)
g.fig.suptitle('Distribution of RoBERTa polarity scores in the reply tweets', fontsize=16, y=1.05)
sns.move_legend(g, "center", bbox_to_anchor=(.55, .45))


# ~~~ Exploring the distribution of scores with plots for Q.3 and post-hoc analysis ~~~


# Grouping the Rai outlets under one single label

rai_group = ['RaiNews', 'Raiofficialnews', 'Tg1Rai', 'tg2rai', 'Tg3web', 'reportrai3', 'agorarai', 'RaiPortaaPorta']

'''Replaces the outlet name with "Rai Group" if it is one of the outlets listed in rai_group. Necessary for meaningful comparison in plots.'''
def sub_rai(df):
    for _, row in df.iterrows():
        if any(x in df.iloc[:,0].tolist() for x in rai_group):
            df.iloc[:,0] = df.iloc[:,0].replace(rai_group, 'RAI Group')
    return df

replies_it = sub_rai(replies_it)
news_it = sub_rai(replies_it)


# RoBERTa scores by outlet

g = sns.displot(data=news_nl, x='RoBERTa_score', hue='RoBERTa_label', col='Outlet', kind='kde')
g.fig.suptitle('Distribution of RoBERTa polarity scores in the Dutch news tweets', fontsize=16, y=1.05)
sns.move_legend(g, "upper left", bbox_to_anchor=(0.075, 0.9))

g = sns.displot(data=replies_nl, x='RoBERTa_score', hue='RoBERTa_label', col='Outlet', kind='kde')
g.fig.suptitle('Distribution of RoBERTa polarity scores in the Dutch replies', fontsize=16, y=1.05)
sns.move_legend(g, "upper left", bbox_to_anchor=(0.075, 0.9))

g = sns.displot(data=news_it, x='RoBERTa_score', hue='RoBERTa_label', hue_order=['negative', 'neutral', 'positive'], col='Outlet', kind='kde', col_wrap=2)
g.fig.suptitle('Distribution of RoBERTa polarity scores in the Italian news tweets', fontsize=16, y=1.05)
sns.move_legend(g, "upper left", bbox_to_anchor=(0.075, 0.9))

g = sns.displot(data=replies_it, x='RoBERTa_score', hue='RoBERTa_label', col='Outlet', kind='kde', col_wrap=2)
g.fig.suptitle('Distribution of RoBERTa polarity scores in the Italian replies', fontsize=16, y=1.05)
sns.move_legend(g, "upper left", bbox_to_anchor=(0.075, 0.9))


# Same for VADER - goes in Appendix

g = sns.displot(data=news_nl, x='VADER_compound_score', col='Outlet', kind='kde', bw_adjust=.5)
g.fig.suptitle('Distribution of VADER compound scores in the Dutch news tweets', fontsize=16, y=1.05)

g = sns.displot(data=replies_nl, x='VADER_compound_score', col='Outlet', kind='kde', bw_adjust=.5)
g.fig.suptitle('Distribution of VADER compound scores in the Dutch replies', fontsize=16, y=1.05)

g = sns.displot(data=news_it, x='VADER_compound_score', col='Outlet', kind='kde', bw_adjust=.5, col_wrap=2)
g.fig.suptitle('Distribution of VADER compound scores in the Italian news tweets', fontsize=16, y=1.05)

g = sns.displot(data=replies_it, x='VADER_compound_score', col='Outlet', kind='kde', bw_adjust=.5, col_wrap=2)
g.fig.suptitle('Distribution of VADER compound scores in the Italian replies', fontsize=16, y=1.05)


# Discussion: plot salience of topic over time by country
from datetime import datetime

'''Transforms column with a date in string format to a datetime object for analysis'''
def to_datetime(df, col):
    df[col] = pd.to_datetime(df[col])
    return df

# News tweets over time by country

to_datetime(news, 'Date')
g = sns.displot(data=news, x='Date', hue='Language', col='Language', kind='kde', bw_adjust=0.25, height=5, aspect=2, legend=False, col_wrap=1, palette='magma')
g.fig.suptitle('Distribution of news tweets over time', fontsize=20, y=1.05)

to_datetime(replies, 'Date') # Same as above for the replies, though not as interesting as the two plots below
g = sns.displot(data=replies, x='Date', hue='Language', col='Language', kind='kde', bw_adjust=0.25, height=5, aspect=2, legend=False, col_wrap=1, palette='magma')
g.fig.suptitle('Distribution of replies over time', fontsize=20, y=1.05)

# These two plots also show the RoBERTa sentiment label

g = sns.displot(data=news, x='Date', hue='RoBERTa_label', col='Language', kind='kde', bw_adjust=0.25, height=5, aspect=2, col_wrap=1)
g.fig.suptitle('Distribution of news by type over time', fontsize=20, y=1.05)
sns.move_legend(g, "upper left", bbox_to_anchor=(0.1, 0.4))

g = sns.displot(data=replies, x='Date', hue='RoBERTa_label', col='Language', kind='kde', bw_adjust=0.25, height=5, aspect=2, col_wrap=1)
g.fig.suptitle('Distribution of replies by type over time', fontsize=20, y=1.05)
sns.move_legend(g, "upper left", bbox_to_anchor=(0.1, 0.4))


# Top 5 news tweets with highest interaction per country
replies_nl['Tweet'].value_counts().head(5)
replies_it['Tweet'].value_counts().head(5)


# ~~~ Model evaluation on hand-annotated subset (n=200) for Q.1 ~~~


'''Gets a random sample of n = 50 rows from a dataframe, to be hand-annotated by a human as an evaluation set'''
def get_sample(df):
    sample = df.sample(n=50, random_state=42) # for reproducibility
    return sample

sample_replies_nl = get_sample(clean_nl)
sample_replies_it = get_sample(clean_it)

sample_news_nl = get_sample(news_nl)
sample_news_it = get_sample(news_it)


# Reading annotated samples back in after hand-labelling and scoring separately with models above
annotated_news_nl = pd.read_csv("./annotated_news_nl.csv", skiprows=1, header=None)
annotated_news_it = pd.read_csv("./annotated_news_it.csv", skiprows=1, header=None)

annotated_replies_nl = pd.read_csv("./annotated_replies_nl.csv", skiprows=1, header=None)
annotated_replies_it = pd.read_csv("./annotated_replies_it.csv", skiprows=1, header=None)


# Defining subsets of interest
an_total = pd.concat([annotated_replies_nl, annotated_replies_it, annotated_news_nl, annotated_news_it], ignore_index=True)

an_it = pd.concat([annotated_replies_it, annotated_news_it], ignore_index=True)
an_nl = pd.concat([annotated_replies_nl, annotated_news_nl], ignore_index=True)
an_news = pd.concat([annotated_news_nl, annotated_news_it], ignore_index=True)
an_replies = pd.concat([annotated_replies_nl, annotated_replies_it], ignore_index=True)


# Evaluation of model performance on different subsets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

'''Takes in a dataframe (or subset defined above) and plots the confusion matrix and metrics for the two models of interest side by side'''
def cf_matrix(df):
    
    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    fig.suptitle('This is the figure title', fontsize=16, y=1.03)
    colnames = ['VADER_label', 'RoBERTa_label']

    for i, colname in enumerate(colnames):
        y_true = df['human_label']
        y_pred = df[colname]
        cm = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'neutral', 'positive'])
        disp.plot(ax=ax[i], xticks_rotation=45, cmap=plt.cm.Blues)
        disp.ax_.set_title(colname)
        print(classification_report(y_true, y_pred, digits=3))
        
    plt.show()

# For example: (Note that the title needs to be redefined with every call)
cf_matrix(an_news)
cf_matrix(an_replies)

# Label counts per set for manuscript
dfs = [annotated_news_nl, annotated_news_it, annotated_replies_nl, annotated_replies_it]
for i, df in enumerate(dfs):
    print(df['human_label'].value_counts())