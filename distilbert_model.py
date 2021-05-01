import ktrain
import os


predictor = ktrain.load_predictor('distilbert')


def get_prediction(x):

    sent = predictor.predict([x])
    return sent[0]