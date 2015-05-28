
import gzip
import os.path as path
import json
import unicodedata
import re

import numpy as np

from dataset._shared import Dataset

thisdir = path.dirname(path.realpath(__file__))
json_file = path.join(thisdir, 'data', 'news.json.gz')
json_full_file = path.join(thisdir, 'data', 'news.full.json.gz')

def normalize_string(text):
    text = unicodedata.normalize('NFKD', text)
    return re.sub('([^\040-\176]|[#%&{}|~:;[\]^*+\-`<>])', '', text)

def space_seperate(text):
    return re.sub('[,.()\'"!_=?@\n]', ' ', text).split()

def preparse():
    all_text = ""
    text_length = 0
    title_length = 0

    with gzip.open(json_file, 'rt') as f:
        for line in f:
            article = json.loads(line)

            text = normalize_string(article['text'])
            text_length = max(text_length, len(text))

            title = normalize_string(article['title'])
            title_length = max(title_length, len(title))

            all_text += text + title

    unique_chars = '\x00' + ''.join(sorted(list(set(all_text))))

    return (text_length, title_length, unique_chars)

# This is a precomputed version of `preparse()`
unique_chars = '\x00 !"$\'(),./0123456789=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ' \
               '\\_abcdefghijklmnopqrstuvwxyz'
text_length = 999
title_length = 197

char2code = {c: i for i, c in enumerate(unique_chars)}

def str_to_code(str):
    # ASCII is 7bit, so there is no reason to use anything higher than
    # signed int8
    return np.asarray([char2code[c] for c in str + '\x00'], dtype='int8')

def dates(items=None):
    data = []
    target = []

    with gzip.open(json_file, 'rt') as f:
        for i, line in enumerate(f):
            article = json.loads(line)
            data.append(np.datetime64(article['date'], 'ms'))
            target.append(0)
            if (items is not None and i + 1 >= items): break

    return Dataset(np.asarray(data), np.asarray(target))

def words(full=False, items=None):
    data = []
    target = []

    with gzip.open(json_full_file if full else json_file, 'rt') as f:
        for i, line in enumerate(f):
            article = json.loads(line)
            data.append(space_seperate(
                normalize_string(article['title'] + ' ' + article['text'])
            ))
            target.append([])
            if (items is not None and i + 1 >= items): break

    return Dataset(data, target)

def letters(items=None):
    data = []
    target = []

    with gzip.open(json_file, 'rt') as f:
        for i, line in enumerate(f):
            article = json.loads(line)
            data.append(str_to_code(normalize_string(article['text'])))
            target.append(str_to_code(normalize_string(article['title'])))
            if (items is not None and i + 1 >= items): break

    return Dataset(data, target, len(unique_chars), max_output_size=title_length)

def fetch(items=None):
    output = []

    with gzip.open(json_file, 'rt') as f:
        for i, line in enumerate(f):
            article = json.loads(line)
            article['id'] = i
            output.append(article)
            if (items is not None and i + 1 >= items): break

    return output
