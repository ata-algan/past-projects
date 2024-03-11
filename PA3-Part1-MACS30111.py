"""


Ata Algan

Basic algorithms module

Algorithms for efficiently counting and sorting distinct 'entities',
or unique values, are widely used in data analysis.
"""

import math
from util import sort_count_pairs

# Task 1.1
def count_tokens(tokens):
    '''
    Counts each distinct token (entity) in a list of tokens.

    Inputs:
        tokens: list of tokens (must be immutable)

    Returns: dictionary that maps tokens to counts
    '''

    new_dict = {}

    for token in tokens:
        if token in new_dict:
            new_dict[token] += 1
        else:
            new_dict[token] = 1

    return new_dict


# Task 1.2
def find_top_k(tokens, k):
    '''
    Find the k most frequently occuring tokens.

    Inputs:
        tokens: list of tokens (must be immutable)
        k: a non-negative integer

    Returns: list of the top k tokens ordered by count.
    '''
    #Error checking (DO NOT MODIFY)
    if k < 0:
        raise ValueError("In find_top_k, k must be a non-negative integer")

    counts = count_tokens(tokens)
    count_tuple = [(token, count) for (token, count) in counts.items()]
    print(count_tuple)
    ordered_tuple = sort_count_pairs(count_tuple)
    return [i[0] for i in ordered_tuple[:k]]


# Task 1.3
def find_min_count(tokens, min_count):
    '''
    Find the tokens that occur *at least* min_count times.

    Inputs:
        tokens: a list of tokens  (must be immutable)
        min_count: a non-negative integer

    Returns: set of tokens
    '''

    #Error checking (DO NOT MODIFY)
    if min_count < 0:
        raise ValueError("min_count must be a non-negative integer")

    counts = count_tokens(tokens)
    count_tuple = [(token, count) for (token, count) in counts.items()]
    return {i[0] for i in count_tuple if i[1] >= min_count}


# Task 1.4

def compute_tf(docs):
    tf_values = []

    for i in docs:
        tkn = count_tokens(i)

        if not i: 
            tf_values.append({})
            continue

        counter = 0
        token_count = []
        temp_tokens = set(i)
        for j in temp_tokens:
            counter = i.count(j)
            token_count.append((j, counter))

        sorted_token_count = sort_count_pairs(token_count)

        f_dash = sorted_token_count[0][1]
        tf = {}
        for key in tkn.keys():
            tf[key] = 0.5 + 0.5 * ((tkn[key] / f_dash))
        
        
        tf_values.append(tf)

    return tf_values

def compute_idf(docs):
    idf_values = {}
    d_abs = len(docs)

    f_docs = [val for value in docs for val in value]

    for term in set(f_docs):
        count_d = 0
        for doc in docs:
            if term in doc:
                count_d += 1
        idf = math.log(d_abs / count_d)
        idf_values[term] = idf

    return idf_values

def compute_td_idf(docs):
    tf_values = compute_tf(docs)
    idf_values = compute_idf(docs)
    
    tfidf_values = []
    for tf_dict in tf_values:
        tfidf = {key: tf_dict[key] * idf_values[key] for key in tf_dict}
        tfidf_values.append(tfidf)

    return tfidf_values

def find_salient(docs, threshold):
    '''
    Compute the salient words for each document.  A word is salient if
    its tf-idf score is strictly above a given threshold.

    Inputs:
      docs: list of list of tokens
      threshold: float

    Returns: list of sets of salient words
    '''
    tdidf_val = compute_td_idf(docs)
    salience_lst = []

    for i in tdidf_val:
        salient_set = set()
        for token, val in i.items():
            if val > threshold:
                salient_set.add(token)
        salience_lst.append(salient_set)

    return salience_lst
            
            
    
   