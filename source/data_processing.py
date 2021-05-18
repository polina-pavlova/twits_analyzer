import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

stop_words = stopwords.words("english")
stemmer = PorterStemmer()


def text_processing(data):
    data = data.apply(lambda tweet: tweet.lower())  # lower register
    data = data.apply(lambda tweet: re.sub(r"@\S+", "", tweet))  # usernames
    data = data.apply(lambda tweet: re.sub(r"https?://\S+", "", tweet))  # urls
    data = data.apply(lambda tweet: re.sub(r"#\S+", "", tweet))  # hashtags
    data = data.apply(lambda tweet: re.sub(r"\S+\.com", "", tweet))  # sites
    data = data.apply(lambda tweet: re.sub(r"[^A-Za-z0-9]", " ", tweet))  # punctuation
    data = data.apply(
        lambda tweet: [word for word in tweet.split(" ") if word not in stop_words]
    )  # delete stop_words
    data = data.apply(
        lambda tweet: [re.compile(r"(.)\1{2,}").sub(r"\1\1", word) for word in tweet]
    )  # delete characters repeated more frequent than 2 times.
    data = data.apply(lambda tweet: [stemmer.stem(word) for word in tweet])  # stemming
    data = data.apply(lambda tweet: [word for word in tweet if word != ""])
    data = data.apply(lambda tweet: " ".join(tweet))
    return data
