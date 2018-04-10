import pickle
import typing

import numpy as np


class Distribution:
    @classmethod
    def from_agents(cls, agents):
        return cls(np.asarray([agent.langs for agent in agents]))

    @classmethod
    def load(cls, load_file: typing.BinaryIO) -> 'Distribution':
        out = pickle.load(load_file)
        if not isinstance(out, cls):
            raise ValueError('Pickle file was not a distribution object')
        return out

    def __init__(self, langs):
        self.langs = langs

    @property
    def mean_features(self):
        return np.mean(self.langs, axis=0)

    @property
    def standard_deviation(self):
        return np.std(self.langs, axis=0)

    @property
    def winning_feature_index(self):
        return np.argmax(self.mean_features)

    def save(self, save_file: typing.BinaryIO):
        """ Saves the agents' language arrays to a file """
        pickle.dump(self, save_file)
