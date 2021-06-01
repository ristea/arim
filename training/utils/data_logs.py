
def save_logs_train(path_to_history, details):
    path_to_history = path_to_history + '/__hystoryTrain__.txt'
    history = open(path_to_history, "a")
    history.write(details + '\n')
    history.close()


def save_logs_eval(path_to_history, details):
    path_to_history = path_to_history + '/__hystoryEval__.txt'
    history = open(path_to_history, "a")
    history.write(details + '\n')
    history.close()


def save_best_stats(path_to_history, details):
    path_to_history = path_to_history + '/__best_stats__.txt'
    history = open(path_to_history, "a")
    history.write(details + '\n')
    history.close()