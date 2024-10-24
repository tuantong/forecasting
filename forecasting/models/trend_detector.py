import numpy as np


def trend_detector(list_of_index, array_of_data, order=1):
    if len(array_of_data) <= 1:
        return (0, 0)
    else:
        result = np.polyfit(list_of_index, list(array_of_data), order)
        slope, beta = result[0], result[1]
        return float(slope), float(beta)
