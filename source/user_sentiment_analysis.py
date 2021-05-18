import pickle
import pandas as pd
from source.data_processing import text_processing
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import termplotlib as tpl
from tensorflow.keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def load_user_data(username):
    user_data = pd.read_csv(f'./users_base/{username}/{username}.csv')
    user_data['Month'] = user_data.date.apply(lambda x: x.split('-')[1])
    user_data.Month = user_data.Month.apply(lambda m: calendar.month_name[int(m)])
    return user_data[["date", "tweet", "Month"]]


def user_sentiment_analysis_log_reg(username):
    log_reg_model = pickle.load(open('./model/log_regression.model', 'rb'))
    user_data = load_user_data(username)
    user_data.tweet = text_processing(user_data.tweet)
    user_predict = log_reg_model.predict(user_data.tweet)
    user_data['prediction'] = pd.Series(user_predict)
    values = {0: 'Negative', 1: 'Positive'}
    user_data['Sentiment'] = user_data['prediction'].map(values)
    return user_data


def user_sentiment_neural_network(username):
    nn_model = load_model('./model/best_model.hdf5', compile=False)
    nn_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
    user_data = load_user_data(username)
    tokenizer = Tokenizer()
    tweets_for_prediction = pad_sequences(tokenizer.texts_to_sequences(user_data.tweet), maxlen=30)
    user_data['predictions'] = nn_model.predict(tweets_for_prediction)
    user_data['Sentiment'] = user_data.predictions.apply(lambda score: 'Positive' if score > 0.5 else 'Negative')
    return user_data


def get_user_data_for_desired_model(username, model='glm'):
    if model == 'glm':
        return user_sentiment_analysis_log_reg(username)
    else:
        return user_sentiment_neural_network(username)


def png_plots(username):
    user_data = get_user_data_for_desired_model(username)
    sns.set()
    plt.figure(num=None, figsize=(16, 10), dpi=300)
    sns.set_style("white")
    sns.catplot(x='Sentiment', kind='count', data=user_data, palette='tab10')
    plt.title(f"@{username}'s tweets emotions distribution")
    plt.savefig(f'./users_base/{username}/{username}_tweets_sentimnent.png', format='png', bbox_inches='tight')

    sns.set()
    plt.figure(num=None, figsize=(10, 10), dpi=300)
    sns.set_style("white")
    sns.color_palette("tab10")
    sns.catplot(x='Month', hue='Sentiment', kind='count', data=user_data, palette='tab10')
    plt.title(f'@{username} tweets emotions per month')
    plt.savefig(f'./users_base/{username}/{username}_tweets_sentiment_per_month.png', format='png', bbox_inches='tight')


def ascii_plots(username):
    user_data = get_user_data_for_desired_model(username)
    counts = pd.value_counts(user_data.Sentiment)
    counts = pd.DataFrame(counts)
    fig = tpl.figure()
    print(f'\n@{username} tweets emotions distribution')
    fig.barh(counts.Sentiment, list(counts.index), force_ascii=False)
    fig.show()
    with open(f"./users_base/{username}/{username}'s_tweets_sentiment_ascii.txt", 'w') as f:
        f.write(fig.get_string())

    print(f'\n@{username} tweets emotions per month')
    counts = user_data.groupby('Month').aggregate({'Sentiment': 'value_counts'})
    fig = tpl.figure()
    fig.barh(counts.Sentiment, list(counts.index), force_ascii=False)
    fig.show()
    with open(f"./users_base/{username}/{username}'s_tweets_sentiment_per_month_ascii.txt", 'w') as f:
        f.write(fig.get_string())
