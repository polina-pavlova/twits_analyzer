import calendar
import pickle  # noqa

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import termplotlib as tpl
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

from source.data_processing import text_processing


def load_user_data(username: str):
    user_data = pd.read_csv(f"./users_base/{username}/{username}.csv")
    user_data["Month"] = user_data.date.apply(lambda x: x.split("-")[1])
    user_data.Month = user_data.Month.apply(lambda m: calendar.month_name[int(m)])
    user_data.tweet = text_processing(user_data.tweet)
    user_data = user_data[["date", "tweet", "Month"]]
    return user_data


def user_sentiment_analysis_log_reg(username: str):
    log_reg_model = pickle.load(open("./model/log_regression.model", "rb"))  # noqa
    user_data = load_user_data(username)
    user_predict = log_reg_model.predict(user_data.tweet)
    user_data["prediction"] = pd.Series(user_predict)
    values = {0: "Negative", 1: "Positive"}
    user_data["Sentiment"] = user_data["prediction"].map(values)
    return user_data


def user_sentiment_neural_network(username: str):
    nn_model = load_model("./model/best_model.hdf5", compile=False)
    nn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    user_data = load_user_data(username)
    tokenizer = Tokenizer()
    tweets_for_prediction = pad_sequences(
        tokenizer.texts_to_sequences(user_data.tweet), maxlen=30
    )
    user_data['predictions'] = nn_model.predict(tweets_for_prediction)
    user_data["Sentiment"] = user_data.predictions.apply(lambda score: "Positive" if score > 0.5 else "Negative")
    return user_data


def get_user_data_for_desired_model(username: str, model: str):
    if model == "glm":
        return user_sentiment_analysis_log_reg(username)
    else:
        return user_sentiment_neural_network(username)


def png_plots(username: str, model: str):
    user_data = get_user_data_for_desired_model(username, model)
    sns.set()
    plt.figure(num=None, figsize=(16, 10), dpi=300)
    sns.set_style("white")
    sns.catplot(x="Sentiment", kind="count", data=user_data, palette="tab10")
    plt.title(f"@{username}'s tweets emotions distribution")
    plt.savefig(
        f"./users_base/{username}/{username}_tweets_sentiment.png",
        format="png",
        bbox_inches="tight",
    )

    sns.set()
    plt.figure(num=None, figsize=(10, 10), dpi=300)
    sns.set_style("white")
    sns.color_palette("tab10")
    sns.catplot(
        x="Month", hue="Sentiment", kind="count", data=user_data, palette="tab10"
    )
    plt.title(f"@{username} tweets emotions per month")
    plt.savefig(
        f"./users_base/{username}/{username}_tweets_sentiment_per_month.png",
        format="png",
        bbox_inches="tight",
    )


def ascii_plots(username: str):
    user_data = get_user_data_for_desired_model(username)
    counts = pd.value_counts(user_data.Sentiment)
    counts = pd.DataFrame(counts)
    fig = tpl.figure()
    print(f"\n@{username} tweets emotions distribution")  # noqa
    fig.barh(counts.Sentiment, list(counts.index), force_ascii=False)
    fig.show()
    with open(
        f"./users_base/{username}/{username}'s_tweets_sentiment_ascii.txt", "w"
    ) as f:
        f.write(fig.get_string())

    print(f"\n@{username} tweets emotions per month")  # noqa
    counts = user_data.groupby("Month").aggregate({"Sentiment": "value_counts"})
    fig = tpl.figure()
    fig.barh(counts.Sentiment, list(counts.index), force_ascii=False)
    fig.show()
    with open(
        f"./users_base/{username}/{username}'s_tweets_sentiment_per_month_ascii.txt",
        "w",
    ) as f:
        f.write(fig.get_string())
