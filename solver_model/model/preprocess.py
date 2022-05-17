import numpy as np

# preprocess raw data and save train and test set
def preprocess(answer_path, puzzel_path):

    puzzel_list = []
    answer_list = []

    rawData = np.genfromtxt('../dataset/sudoku_dataset.csv', dtype='S81', delimiter=',', skip_header=1)

    for row in rawData:
        puzzel = np.frombuffer(row[0], np.int8) - 48
        puzzel = puzzel.reshape((9, 9))
        puzzel_list.append(puzzel)

        answer = np.frombuffer(row[1], np.int8) - 48
        answer = answer.reshape((9, 9))
        answer_list.append(answer)

    np.save(puzzel_path, np.array(puzzel_list))
    np.save(answer_path, np.array(answer_list))





