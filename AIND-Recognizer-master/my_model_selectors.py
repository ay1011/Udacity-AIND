import math
import statistics
import operator
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states,
                                    covariance_type="diag",
                                    n_iter=1000,
                                    random_state=self.random_state,
                                    verbose=False)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model.fit(self.X, self.lengths)
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def argmin(dictionary):
        v = list(dictionary.values())
        k = list(dictionary.keys())
        return k[v.index(min(v))]

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        bic_scores = []
        n_components = range(self.min_n_components, self.max_n_components + 1)

        N, n_features = self.X.shape

        for n_component in n_components:
            try:
                model = self.base_model(n_component)
                logL  = model.score(self.X, self.lengths)
                #Parameters = Initial state occupation probabilities: n_component*(n_component - 1)
                #           + Transition probabilities: n_component-1
                #           + Emission probabilities (numMeans+numCovars) : 2*n_component*n_features
                p = n_component*n_component - 1 + 2*n_component*n_features
                logN = np.log(N)

                bic_score = -2*logL + p*logN

                # Add bic_score to scores list
                bic_scores.append(bic_score)

            except:
                pass

        # Return best model based on BIC
        best_n_components = n_components[np.argmin(bic_scores)] if bic_scores else self.n_constant
        return self.base_model(best_n_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on DIC scores


        logLs = []
        n_components = range(self.min_n_components, self.max_n_components + 1)

        for n_component in n_components:
            try:
                hmm_model           = self.base_model(n_component)
                # 1. Get logL values for all words
                logL = hmm_model.score(self.X, self.lengths)
                logLs.append(logL)

            except:
                pass

        # 2. Implement DIC formula
        M = len(n_components)
        sum_logLs = sum(logLs)

        dic_scores = []
        for logL in logLs:
            # DIC = likelihood(this word) - average likelihood(other words)
            dic_score = logL - ((sum_logLs - logL) / (M - 1))

            # Add dic_score to scores list
            dic_scores.append(dic_score)

        # Return best model based on DIC
        best_n_components = (n_components[np.argmax(dic_scores)] or n_components[0]) if dic_scores else self.n_constant
        return self.base_model(best_n_components)


class SelectorCV(ModelSelector):
    '''
    Select best model based on average log Likelihood of cross-validation folds
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection using CV

        scores = []
        split_method = KFold()
        n_components = range(self.min_n_components, self.max_n_components + 1)

        for n_component in n_components:
            try:
                model = self.base_model(n_component)

                # If splitting is not possible
                if len(self.sequences) < 2:
                    # Add scores mean to scores list
                    scores.append(np.mean(model.score(self.X, self.lengths)))

                else:
                    test_scores = []

                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        # Setup training sequences
                        self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                        test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                        test_scores.append(model.score(test_X, test_lengths))

                    # Add test_scores mean to scores list
                    scores.append(np.mean(test_scores))

            except:
                pass

        # Return best model
        best_n_components = n_components[np.argmax(scores)] if scores else self.n_constant
        return self.base_model(best_n_components)