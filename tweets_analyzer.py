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
        help="Type of bar charts with user statistics. Default - png.",
        required=False,
        default="png",
        choices=["png", "ascii"],
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        help="Choose desired model: glm (logistic regression model) or nn (LSTM neural network). Default - glm.",
        required=False,
        default="glm",
        choices=["glm", "nn"],
        type=str,
    )
    parser.add_argument(
        "-l",
        "--users_list",
        dest="users_in_base",
        help="Print list of users presented in base. Default - False.",
        required=False,
        type=bool,
        default=False,
        choices=[True, False],
    )

    return parser.parse_args()


def user_analysis(username: str, plot: str, model: str, users: list):
    if username not in users:
        os.mkdir(f"./users_base/{username}")

    if model in os.listdir(f"./users_base/{username}"):  # username in users and
        print(
            f"@{username}'s tweets analysis with {model} is in folder users_base/{username}/{model}"
        )
    else:
        os.mkdir(f"./users_base/{username}/{model}")
        download_user_tweets(username)
        print(f"Draw charts with @{username} sentiment analysis\n")
        if plot == "ascii":
            ascii_plots(username, model)
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
    users = os.listdir("./users_base/")

    if users_in_base:
        print("Users below are in base:")
        print("\n".join(users))  # noqa

    if not username:
        raise ArgumentError(
            f"Username in twitter is needed for semantics analysis. Following users are presented in users_base folder: {users}"
        )

    else:
        user_analysis(username, plot, model, users)


if __name__ == "__main__":
    print(
        """
    ##############################################
    #              Tweets sentiment              #
    #                  analysis                  #
    ##############################################\n
    """
    )
    main()
