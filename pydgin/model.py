import typing

import numpy as np
from numba import jit

from pydgin import analysis

black = 0
european = 1
mixed = 2


class Population:
    """ Akin to an agent, we use a population because it is faster to manipulate agents in aggregate when using numpy. """

    def __init__(
            self, *,
            races: np.ndarray,
            native_langs: np.ndarray,
            n_langs: int,
            lr_coordinated: float,
            lr_uncoordinated: float,
            select: typing.Callable[['Population'], typing.Tuple[np.ndarray, np.ndarray]],
            non_learning_europeans: bool,
    ):
        self.non_learning_europeans = non_learning_europeans
        assert races.shape == native_langs.shape
        self.races: np.ndarray = races
        self.lr_coordinated = lr_coordinated
        self.lr_uncoordinated = lr_uncoordinated
        self.native_languages = None
        self.langs = None
        self.set_native_languages(native_langs, n_langs)
        self.select = select

    @property
    def population_size(self):
        return self.langs.shape[0]

    @property
    def n_langs(self):
        return self.langs.shape[1]

    def set_native_languages(self, new_langs, n_langs):
        self.native_languages = new_langs
        self.langs = np.zeros(shape=(new_langs.size, n_langs), dtype=np.float)
        self.langs[np.arange(self.population_size), new_langs] = 1.0

    def merge(self, other: 'Population'):
        """ Add another population to this one. """
        self.replace_merge(self, other)

    def replace_merge(self, p1, p2):
        """ Replace this population's constituents ("agents") with the combination of two other populations. """
        self.langs = np.concatenate((p1.langs, p2.langs))
        self.races = np.concatenate((p1.races, p2.races))
        self.native_languages = np.concatenate((p1.native_languages, p2.native_languages))
        assert self.langs.shape[0] == self.races.shape[0] == self.native_languages.shape[0]

    def chat_amongst_yourselves(self):
        us_idx, them_idx = self.select(self)
        chatter(self.langs,
                self.races,
                us_idx,
                them_idx,
                self.lr_coordinated,
                self.lr_uncoordinated,
                self.non_learning_europeans,
                )

    def kill(self, lang_index, lang_delta):
        group_idx = np.arange(self.population_size)[self.native_languages == lang_index]
        kill_idx = np.random.choice(group_idx, size=np.abs(lang_delta), replace=False)
        self.races = np.delete(self.races, kill_idx, axis=0)
        self.langs = np.delete(self.langs, kill_idx, axis=0)
        self.native_languages = np.delete(self.native_languages, kill_idx, axis=0)


# =========================================================================================
# Numba-compiled tight loops
# -----------------------------------------------------------------------------------------
# Numba's ability to compile classes is limited at the moment so here we pull out the major
# slow parts of the model into distinct functions that are called by the Population class.

@jit(nopython=True)
def speak(langs: np.ndarray):
    r = np.random.random()
    return np.argmax(langs.cumsum() > r)


@jit(nopython=True)
def update(population, races, idx, my_lang, your_lang, lrc, lru, non_learning_europeans):
    """ Fast update a single individual a-la the java version. """
    if non_learning_europeans and races[idx] == european:
        return
    learning = lrc if my_lang == your_lang else lru
    population[idx][your_lang] += learning * (1 - population[idx][your_lang])

    for i in range(len(population[idx])):
        if i != your_lang:
            population[idx][i] -= learning * population[idx][i]


@jit(nopython=True)
def chatter(population, races, us_idx, them_idx, lr_coordinated, lr_uncoordinated, non_learning_europeans):
    for me_idx, you_idx in zip(us_idx, them_idx):
        my_lang = speak(population[me_idx])
        your_lang = speak(population[you_idx])
        update(population, races, me_idx, my_lang, your_lang, lr_coordinated, lr_uncoordinated, non_learning_europeans)
        update(population, races, you_idx, your_lang, my_lang, lr_coordinated, lr_uncoordinated, non_learning_europeans)
# =========================================================================================


class PopulationFactory:
    def __init__(self, *, lr_coordinated: float, lr_uncoordinated: float, select, non_learning_europeans, features,
                 non_white):
        self.non_learning_europeans = non_learning_europeans
        self.select = select
        self.lr_coordinated = lr_coordinated
        self.lr_uncoordinated = lr_uncoordinated

        self.features = features
        self.n_langs = np.max(features) + 1
        self.non_white = non_white

    def lookup_group(self, group_index):
        """ Features contains MC in the first column so we skip. """
        return self.features[group_index + 1]

    def __call__(self, *, races, native_langs) -> Population:
        return Population(
            races=races,
            native_langs=native_langs,
            n_langs=self.n_langs,
            lr_coordinated=self.lr_coordinated,
            lr_uncoordinated=self.lr_uncoordinated,
            select=self.select,
            non_learning_europeans=self.non_learning_europeans,
        )

    def get_empty(self) -> Population:
        return self(
            races=np.empty((0,), dtype=np.int),
            native_langs=np.empty((0,), dtype=np.int),
        )

    def make_n(self, n, group_index) -> Population:
        native_lang = self.lookup_group(group_index)
        race = european if group_index == 0 else self.non_white
        return self(
            races=np.full((n,), race, dtype=np.int),
            native_langs=np.full((n,), native_lang, dtype=np.int)
        )


def intseed(seed):
    if isinstance(seed, str):
        return int(''.join(str(ord(c) - ord('a')) for c in seed))
    return seed


class Model:
    def __init__(self, *,
                 population_deltas,
                 epoch_lengths,
                 select,
                 lr_coordinated,
                 lr_uncoordinated,
                 non_learning_europeans,
                 plantation_epoch,
                 randomseed=None
                 ):
        self.population_deltas = population_deltas
        self.epoch_lengths = epoch_lengths
        self.select = select
        self.seed = intseed(randomseed)
        self.lr_coordinated = lr_coordinated
        self.lr_uncoordinated = lr_uncoordinated
        self.non_learning_europeans = non_learning_europeans
        self.plantation_epoch = plantation_epoch  # The point at which the plantation phase starts

    def run(self, feature_columns):
        # Initialisation
        np.random.seed(self.seed)
        agent_factory = PopulationFactory(lr_coordinated=self.lr_coordinated,
                                          lr_uncoordinated=self.lr_uncoordinated,
                                          select=self.select,
                                          non_learning_europeans=self.non_learning_europeans,
                                          features=feature_columns,
                                          non_white=mixed if self.plantation_epoch else black,
                                          )
        population = agent_factory.get_empty()
        days_passed = 0

        for epoch_popdeltas, days_in_epoch in zip(self.population_deltas, self.epoch_lengths):
            # Dis waar die rasse klassifikasie gekies word. Die opsies is rise ranks of nie, as nie en die plantasie
            # periode het nog nie begin nie, dan ...
            if self.plantation_epoch and agent_factory.non_white != mixed and days_passed >= self.plantation_epoch:
                agent_factory.non_white = black

            # Immigration / Death
            for group_index, group_delta in enumerate(epoch_popdeltas):
                if group_delta > 0:
                    immigrants = agent_factory.make_n(group_delta, group_index)
                    population.merge(immigrants)
                elif group_delta < 0:
                    population.kill(agent_factory.lookup_group(group_index), group_delta)

            # Communication and passing of time
            days_passed += days_in_epoch
            if population.population_size == 1:
                # Skip communication if there's only one person; we do this outside
                # of the communicate loop because the population size isn't going to
                # change in there
                continue
            for _ in range(days_in_epoch):
                population.chat_amongst_yourselves()
        return analysis.Distribution(population.langs)
