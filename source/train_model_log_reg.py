import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, plot_roc_curve
import matplotlib.pyplot as plt

import pickle
from data_processing import text_processing


def process_data(data):
    data.rename(columns={0: 'polarity', 5: 'text'}, inplace=True)
    data['polarity'] = data['polarity'].map({0: 0, 4: 1})
    data.text = text_processing(data.text)
    return data


def load_data(path_to_data):
    data = pd.read_csv(path_to_data, encoding='latin', header=None)
    data = data[[0, 5]]
    return process_data(data)


def log_regression_model(data):
    x_train, x_test, y_train, y_test = train_test_split(data.text, data.polarity, test_size=0.2,
                                                      stratify=data.polarity, random_state=42)
    word_vectorizer = TfidfVectorizer()
    classifier = LogisticRegression(max_iter=2000000)
    pipeline = Pipeline([('vect', word_vectorizer), ('clf', classifier)])
    pipeline.fit(x_train, y_train)

    pipe_pred = pipeline.predict(x_test)
    cr = classification_report(y_test, y_pred=pipe_pred)
    with open('../model/log_reg_metrics.txt', 'w') as file:
        file.write(cr)

    plot_roc_curve(pipeline, x_test, y_test)
    plt.savefig('../model/roc_auc_log_reg.png', format='png')
    pickle.dump(pipeline, open("../model/log_regression.model", 'wb'))


twitter_data = load_data('../data/data.csv')
log_regression_model(twitter_data)
