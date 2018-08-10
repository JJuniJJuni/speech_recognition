from keras.models import model_from_json

import numpy as np

from preprocess import get_labels
from preprocess import wav2mfcc


def save_model(neural_model):  # save CNN model to json
    # serialize model to JSON
    model_json = neural_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    neural_model.save_weights("model.h5")
    print("Saved model to disk")
    # later...


def load_model():  # load json to CNN model
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model


def predict(filepath, model=None):  # predict english word based CNN
    sample = wav2mfcc(filepath)
    feature_dim_1, feature_dim_2, channel = 20, 11, 1
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]


if __name__ == '__main__':
    model = load_model()
    print("after loading, predicted word:",
          predict('./data/training/cat/0c5027de_nohash_0.wav', model=model))
