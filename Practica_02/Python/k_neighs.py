from numpy.typing import NDArray
import numpy as np
import pandas as pd

# region Types

NPArray = NDArray[np.int8]

# endregion


def read_data(src, target_column):
    data = pd.read_csv(src)
    target_column_index = data.columns.get_loc(target_column)
    return data.values, target_column_index


def euclidean_distance(x: NPArray, y: NPArray):
    return np.sqrt(np.sum(np.square(x - y)))


def get_neighbors(train, test_row, num_neighbors):
    distances = [
        (train_row, euclidean_distance(test_row[1:], train_row[1:]))
        for train_row in train
    ]
    distances.sort(key=lambda tup: tup[1])
    neighbors = [distances[i][0] for i in range(num_neighbors)]
    return neighbors


def predict(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [neigh[0] for neigh in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


def main():
    INPUT_FILE = "./heart_2020_small_sanitazed.csv"
    TARGET_COLUMN = "HeartDisease"
    K_NEIGHBORS = 3

    data, target_column_index = read_data(INPUT_FILE, TARGET_COLUMN)
    test_rows, train_rows = data[:50], data[50:]

    predictions = []
    for test_row in test_rows:
        prediction = predict(train_rows, test_row, num_neighbors=K_NEIGHBORS)
        predictions.append(prediction)

    predictions_df = pd.DataFrame(
        data=predictions,
        columns=["prediction"],
    )


if __name__ == "__main__":
    main()
