import os

from predict import load_model
from predict import predict

from tests import TEST_PATH

from tqdm import tqdm

TRAINGING_PATH = "./data/training"


def get_training_set_counts(path=TRAINGING_PATH):
    labels = os.listdir(path)
    counts_dict = {}
    for label in labels:
        counts_dict[label] = len(os.listdir(path + '/' + label))
    return counts_dict


def get_accuracy(path=TEST_PATH, model=load_model()):
    labels = os.listdir(path)
    percent_dict = {}
    for label in tqdm(labels):
        label_path = path + '/' + label
        sounds = os.listdir(label_path)
        all_counts, correctd_counts = len(sounds), 0
        for sound in tqdm(sounds):
            if predict(label_path + '/' + sound, model) == label:
                correctd_counts += 1
        percent_dict[label] = round((100 * correctd_counts/all_counts), 2), all_counts
    return percent_dict


training_set = get_training_set_counts()
result = get_accuracy()
for key in result:
    print("{0}'s accuracy: {1} (training_set:{2} test_set:{3})"
          .format(key, result[key][0], training_set[key], result[key][1]))
