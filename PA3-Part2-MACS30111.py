"""
Ata Algan

Analyze module

Functions to analyze tweets. 
"""

import unicodedata
import sys

from basic_algorithms import find_top_k, find_min_count, find_salient

##################### DO NOT MODIFY THIS CODE #####################

def keep_chr(ch):
    '''
    Find all characters that are classifed as punctuation in Unicode
    (except #, @, &) and combine them into a single string.
    '''
    return unicodedata.category(ch).startswith('P') and \
        (ch not in ("#", "@", "&"))

PUNCTUATION = " ".join([chr(i) for i in range(sys.maxunicode)
                        if keep_chr(chr(i))])

# When processing tweets, ignore these words
STOP_WORDS = ["a", "an", "the", "this", "that", "of", "for", "or",
              "and", "on", "to", "be", "if", "we", "you", "in", "is",
              "at", "it", "rt", "mt", "with"]

# When processing tweets, words w/ a prefix that appears in this list
# should be ignored.
STOP_PREFIXES = ("@", "#", "http", "&amp")


#####################  MODIFY THIS CODE #####################


############## Part 2 ##############

# Task 2.1
def find_top_k_entities(tweets, entity_desc, k):
    '''
    Find the k most frequently occuring entitites.

    Inputs:
        tweets: a list of tweets
        entity_desc: a triple such as ("hashtags", "text", True),
          ("user_mentions", "screen_name", False), etc.
        k: integer

    Returns: list of entities
    '''

    tweet_tokens = []

    for tweet in tweets:
        for j in tweet["entities"][entity_desc[0]]:
            if entity_desc[2]:
                tweet_tokens.append(j[entity_desc[1]])
            else:
                tweet_tokens.append(j[entity_desc[1]].lower())

    return find_top_k(tweet_tokens, k)


# Task 2.2
def find_min_count_entities(tweets, entity_desc, min_count):
    '''
    Find the entitites that occur at least min_count times.

    Inputs:
        tweets: a list of tweets
        entity_desc: a triple such as ("hashtags", "text", True),
          ("user_mentions", "screen_name", False), etc.
        min_count: integer

    Returns: set of entities
    '''

    tweet_tokens = []

    for i in tweets:
        for j in i["entities"][entity_desc[0]]:
            if entity_desc[2]:
                tweet_tokens.append(j[entity_desc[1]])
            else:
                tweet_tokens.append(j[entity_desc[1]].lower())

    return find_min_count(tweet_tokens, min_count)



############## Part 3 ##############

# Pre-processing step and representing n-grams

def preparation(tweet, case_sensitive, remove_stop):
    new_tweets = []
    abridged_txt = tweet["abridged_text"].split()
 
    for word in abridged_txt:
        #Remove any leading and trailing punctuation from each word. 
        word = word.strip(PUNCTUATION)
        #For tasks that are not case sensitive, convert the word to lower case.
        if not case_sensitive:
            word = word.lower()
        #For the tasks that require it, eliminate all stop words (case sensitive)
        if remove_stop:
            if word in STOP_WORDS:
                word = word.replace(word, "")
        #Remove URLs, hashtags, and mentions.
        if word.startswith(STOP_PREFIXES):
            word = ""

        if word != "":
            new_tweets.append(word)

    return new_tweets


# Task 3.1
def find_top_k_ngrams(tweets, n, case_sensitive, k):
    '''
    Find k most frequently occurring n-grams.

    Inputs:
        tweets: a list of tweets
        n: integer
        case_sensitive: boolean
        k: integer

    Returns: list of n-grams
    '''

    top_k_lst = []
    for i in tweets:
        prepared_tweet = preparation(i, case_sensitive, True)
        ngrams = [tuple(prepared_tweet[i:i+n]) for i in range(0, len(prepared_tweet) - n + 1)]
        top_k_lst.extend(ngrams)

    top_k_lst_new = find_top_k(top_k_lst,k)

    return top_k_lst_new


# Task 3.2
def find_min_count_ngrams(tweets, n, case_sensitive, min_count):
    '''
    Find n-grams that occur at least min_count times.

    Inputs:
        tweets: a list of tweets
        n: integer
        case_sensitive: boolean
        min_count: integer

    Returns: set of n-grams
    '''
    min_count_lst = []
    for i in tweets:
        prepared_tweet = preparation(i, case_sensitive, True)
        ngrams = [tuple(prepared_tweet[i:i+n]) for i in range(0, len(prepared_tweet) - n + 1)]
        min_count_lst.extend(ngrams)

    set_ngrams = find_min_count(min_count_lst,min_count)

    return set(set_ngrams)


# Task 3.3
def find_salient_ngrams(tweets, n, case_sensitive, threshold):
    '''
    Find the salient n-grams for each tweet.

    Inputs:
        tweets: a list of tweets
        n: integer
        case_sensitive: boolean
        threshold: float

    Returns: list of sets of strings
    '''

    salient_tweets = []
    for i in tweets:
        prepared_tweet = preparation(i, case_sensitive, False)
        ngrams = [tuple(prepared_tweet[i:i+n]) for i in range(0, len(prepared_tweet) - n + 1)]
        salient_tweets.append(ngrams)
        print(salient_tweets)

    salient_list = find_salient(salient_tweets,threshold)

    return salient_list
