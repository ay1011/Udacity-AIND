import warnings
from asl_data import SinglesData
import math
import operator

def _score_data(model, X, y):
    try:
        return model.score(X, y)
    except:
        return -math.inf

def argmax(dictionary):
    v = list(dictionary.values())
    k = list(dictionary.keys())
    return k[v.index(max(v))]

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer

    for X, y in test_set.get_all_Xlengths().values():
        seq_probs = {word: _score_data(model, X, y) for word, model in models.items()}
        probabilities.append(seq_probs)
        guesses.append(argmax(seq_probs))
    return probabilities, guesses