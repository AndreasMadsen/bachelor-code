
import plot
import dataset

import os.path as path

articles = dataset.news.fetch(3)

for article in articles:
    print('title: ')
    print(article['title'])

    print('text: ')
    print(article['text'])
