
import os
from source.user_sentiment_analysis import png_plots, ascii_plots
from source.get_user_tweets import download_user_tweets

username = 'tim_cook'
os.mkdir(f"./users_base/{username}")
download_user_tweets(username)

ascii_plots(username)
