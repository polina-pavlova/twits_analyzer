import twint
import aiohttp


def download_user_tweets(username, since='2021-1-1'):
    print(f'Downloading @{username} tweets\n')
    c = twint.Config()
    c.Username = username
    c.Since = since
    c.Store_csv = True
    c.Output = f"./users_base/{username}/{username}.csv"
    c.Hide_output = True
    try:
        twint.run.Search(c)
    except aiohttp.client_exceptions.ClientOSError:
        print("Connection reset by peer")
