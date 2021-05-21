import subprocess


def subprocess_for_test(command):
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process.communicate()


def test_username_and_list_of_users_at_same_time():
    cmd = 'python3 tweets_analyzer.py -u username -l'
    output, error = subprocess_for_test(cmd)
    assert "argument -l/--users_list: not allowed with argument -u/--username" in error.decode('utf-8')


def test_output_for_user_presented_in_base():
    cmd = 'python3 tweets_analyzer.py -u twitter'
    output, error = subprocess_for_test(cmd)
    assert "@twitter's tweets analysis with glm is in folder users_base/twitter/glm" in output.decode('utf-8')

