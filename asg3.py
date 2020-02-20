#!/usr/bin/env python
"""NLP Preprocessing Library"""
import nltk
import zipfile
import os
import pandas as pd
import json

file_is_read = False
mode = ""
corpus = []


def clean_text(raw_text):
    """Remove url, tokens"""
    raw_text = raw_text.replace('RT ', '')
    text_list = raw_text.split()
    useless_tokens = ["@", "#"]
    text_list = [x for x in text_list if x[0] not in useless_tokens]
    text_list = [x for x in text_list if x.lower().find('http://') == -1]
    text_list = [x for x in text_list if x.lower().find('https://') == -1]
    text = " ".join(text_list)
    # remove other tokens
    return text

def tokenize_text(tweet_str):
    """convert string to chunks of text"""
    tokenizer = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)
    return tokenizer.tokenize(tweet_str)
    # Source: https://www.nltk.org/api/nltk.tokenize.html


def process_dictionary(_list):
    """remove unknown or unk tag, insert pad and unknown tag"""
    try:
        del _list[_list.index('<unknown>')]
    except ValueError:
        pass

    try:
        del _list[_list.index('<unk>')]
    except ValueError:
        pass
    _list.insert(0, '<pad>')
    _list.insert(0, '<unknown>')

    return _list


def replace_token_with_index(tokenized_tweet, max_length_dictionary, file_path="dict.zip/dict/Glove_dict.txt"):
    """convert each text to dictionary index"""

    # should be replacing each token in a list of tokens by their corresponding index
    # Source: https://github.com/stanfordnlp/GloVe
    # Source: https://towardsdatascience.com/word-embeddings-for-sentiment-analysis-65f42ea5d26e

    global file_is_read
    global mode
    global corpus

    if file_is_read:
        if mode == "zip":
            for idx, word in enumerate(tokenized_tweet):
                if word not in corpus:
                    tokenized_tweet[idx] = "<unknown>"
        else:
            for idx, word in enumerate(tokenized_tweet):
                if word not in corpus:
                    tokenized_tweet[idx] = "<unk>"
        return [corpus.index(x) for x in tokenized_tweet]

    corpus = [""]*max_length_dictionary

    # asg3.replace_token_with_index(["this", "is", "tweet"], 
    # 10000000, "Archive.zip/dict/Glove_dict.txt")
    if ".zip/" in file_path:
        archive_path = os.path.abspath(file_path)
        split = archive_path.split(".zip/")
        archive_path = split[0] + ".zip"
        path_inside = split[1]
        archive = zipfile.ZipFile(archive_path, "r")
        embeddings = archive.read(path_inside).decode("utf8").split("\n")

        embeddings = process_dictionary(embeddings)

        idx = 0
        for word in embeddings:
            corpus[idx] = word
            idx += 1

        for idx, word in enumerate(tokenized_tweet):
            if word not in corpus:
                tokenized_tweet[idx] = "<unknown>"

        file_is_read = True
        mode = "zip"

        return [corpus.index(x) for x in tokenized_tweet]
    else:
        file = open(file_path, "r")
        idx = 0
        for word in file:
            corpus[idx] = word.rstrip()
            idx += 1

        for idx, word in enumerate(tokenized_tweet):
            if word not in corpus:
                tokenized_tweet[idx] = "<unk>"

        file_is_read = True
        mode = "txt"

        return [corpus.index(x) for x in tokenized_tweet]



def pad_sequence(arr, max_length_tweet):
    """add 0 padding to the trail until max_length_tweet"""
    # padding a list of indices with 0 until a maximum length (max_length_tweet)
    trailing_zeros = [0]*(max_length_tweet-len(arr))
    arr.extend(trailing_zeros)
    return arr



def get_glove_dictionary(file_path="./glove.twitter.27B.25d.txt"):
    """output a glove dictionary"""
    file = open(file_path, "r")
    dictionary = {}
    keys = []
    for word_vector in file:
        dictionary[word_vector.split()[0]] = word_vector.split()[1:]
        keys.append(word_vector.split()[0])
    file.close()

    file = open("./Glove_dict.txt", "a")
    for word in keys:
        file.write(word + '\n')
    file.close()


def preprocess_text(input_text, max_length_tweet=100000, max_length_dictionary=1193516):
    """a general method to call, convert string to vectorized representation"""
    input_text = clean_text(input_text)
    text_list = tokenize_text(input_text)
    index_list = replace_token_with_index(text_list, max_length_dictionary)
    index_list = pad_sequence(index_list, max_length_tweet)
    return index_list

def preprocss_file(file_path):
    df = pd.read_csv(file_path, encoding='windows-1252')

    result = []
    for idx, row in df.iterrows():
        print(idx)
        _dict = {}
        _dict["features"] = preprocess_text(row["Tweet"], max_length_tweet = 140)
        _dict["sentiment"] = row["Sentiment"]
        result.append(_dict)


    output_file_name = file_path.split(".")[0] + '.json'
    with open(output_file_name, 'w', encoding='utf-8') as f:
        for entry in result:
            json.dump(entry, f)
            f.write('\n')
    f.close()

def generate_json():
    preprocss_file("eval.csv")
    preprocss_file("train.csv")
    preprocss_file("dev.csv")

# generate_json()