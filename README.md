# Bachelor Thesis (Code)

_By: Andreas Madsen_

## Download

```shell
git clone https://github.com/AndreasMadsen/bachelor-code.git code
```

## Run code

check the `run` directory for executabel scripts. Note that the
articles aren’t inclided as I don’t have the interlectual property
rights to share them.

## Dataset format

There are two datasets, `dataset/data/news.full.json.gz` and `dataset/data/news.json.gz`. They contain pretty much the same thing, but some methods only used a subset of the article text, thus they used the `news.json.gz` file.

The format is gziped newline seperated json strings. The JSON format is:

```javascript
{
  "title": /* title as unicode text, shouldn't contain newlines */,
  "text": /* main text as unicode text, may contain newlines */,
  "website": /* website index, arbitrary number */,
  "date": /* unix timestamp in ms */,
  "href": /* http url */
}
```
