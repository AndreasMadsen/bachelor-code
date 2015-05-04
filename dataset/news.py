
import gzip
import os.path as path
import json
import unicodedata
import re

thisdir = path.dirname(path.realpath(__file__))
json_file = path.join(thisdir, 'data', 'news.json.gz')

def normalize_string(text):
    text = unicodedata.normalize('NFKD', text)
    return re.sub('[^\040-\176]', '', text)

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

    unique_chars = ''.join(sorted(list(set(all_text))))

    return (text_length, title_length, unique_chars)

# This is a precomputed version of `preparse()`
unique_chars = "\x00 !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
               "[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
text_length = 48112
title_length = 197

char_2_code = {c: i for i, c in enumerate(unique_chars)}

def str_to_code(str):
    # ASCII is 7bit, so there is no reason to use anything higher than
    # signed int8
    return np.asarray([char_2_code[c] for c in str + '\x00'], dtype='int8')

def news():
    data = []
    target = []

    with gzip.open(json_file, 'rt') as f:
        for line in f:
            article = json.loads(line)
            data.append(str_to_code(normalize_string(article.text)))
            target.append(str_to_code(normalize_string(article.title)))

    return Dataset(data, target, len(unique_chars))
