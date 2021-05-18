import aiohttp
import twint


def download_user_tweets(username: str):
    print(f"Downloading @{username} tweets\n")  # noqa
    c = twint.Config()
    c.Username = username
    c.Since = "2021-1-1"
    c.Store_csv = True
    c.Output = f"./users_base/{username}/{username}.csv"
    c.Hide_output = True
    try:
        twint.run.Search(c)
    except aiohttp.client_exceptions.ClientOSError:
        print("Connection reset by peer")  # noqa
