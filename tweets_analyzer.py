import argparse
import os

from source.get_user_tweets import download_user_tweets
from source.user_sentiment_analysis import ascii_plots, png_plots


class ArgumentError(Exception):
    """
    Username in twitter is required for sentiment analysis
    """


def make_arguments_parser():
    parser = argparse.ArgumentParser(
        description="Application for sentiment analysis of tweets published since the beginning of 2021"
    )

    parser.add_argument(
        "-u",
        "--username",
        dest="username",
        help="Twitter username for sentiment analysis",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-p",
        "--plot",
        dest="plot",
        help="Type of bar charts with user statistics (png/ascii). Default - ascii.",
        required=False,
        default="png",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        help="Choose desired model: glm (for logistic regeression model) or nn (for LSTM neural network model). Default - glm.",
        required=False,
        default="glm",
        type=str,
    )
    parser.add_argument(
        "-l",
        "--users_list",
        dest="users_in_base",
        help="Print list of users presented in base (True/False). Default - False.",
        required=False,
        type=bool,
        default=False,
    )

    return parser.parse_args()


def user_analysis(username: str, plot: str, model: str):
    os.mkdir(f"./users_base/{username}")
    download_user_tweets(username)
    print(f"Draw charts with @{username} sentiment analysis\n")
    if plot == "ascii":
        ascii_plots(username)
    png_plots(username, model)
    print(
        f"You can find all plots with @{username} sentiment analysis in users_base/{username} folder"
    )


def main():
    args = make_arguments_parser()
    username = args.username
    plot = args.plot
    model = args.model
    users_in_base = args.users_in_base

    if users_in_base:
        print("Users below are in base:")
        print("\n".join(users))  # noqa

    if not username:
        raise ArgumentError(
            f"Username in twitter is needed for semantics analysis. Following users are presented in users_base folder: {users}"
        )

    elif username in users:
        print(
            f"@{username}'s tweets analysis with all charts are in folder users_base/{username}"
        )

    else:
        user_analysis(username, plot, model)

if __name__ == "__main__":
    print(
        """
    ##############################################
    #              Tweets sentiment              #
    #                  analysis                  #
    ##############################################\n
    """
    )
    users = os.listdir("./users_base/")
    main()
