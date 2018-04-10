import numpy as np

from pydgin import model


def random(population):
    us = np.arange(population.population_size)
    them = np.random.choice(us, size=us.size, replace=True)
    while np.any(us == them):
        # Madmen are talking to themselves
        madmen_bidx = us == them
        them[madmen_bidx] = np.random.choice(us, size=np.count_nonzero(madmen_bidx), replace=True)
    return us, them


class ColdShoulder:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def __call__(self, population: model.Population):
        us_idx, them_idx = random(population)

        # (white, black) pair
        our_whites = population.races[us_idx] == model.european
        their_blacks = population.races[them_idx] == model.black
        random_numbers = np.random.random(size=our_whites.shape)
        cold_idx = np.logical_and(np.logical_and(our_whites, their_blacks), random_numbers > self.epsilon)

        # (black, white) pair
        our_blacks = population.races[us_idx] == model.black
        their_whites = population.races[them_idx] == model.european
        random_numbers = np.random.random(size=our_blacks.shape)
        cold_idx2 = np.logical_and(np.logical_and(our_blacks, their_whites), random_numbers > self.epsilon)

        warm_idx = np.logical_not(np.logical_or(cold_idx, cold_idx2))
        return us_idx[warm_idx], them_idx[warm_idx]

    @property
    def __name__(self):
        return f'{self.__class__.__name__}({self.epsilon})'


class Segregation:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def __call__(self, population: model.Population):
        us_idx, them_idx = random(population)

        # (white, black) pair
        our_whites = population.races[us_idx] == model.european
        their_blacks = population.races[them_idx] == model.black
        random_numbers = np.random.random(size=our_whites.shape)
        to_be_replaced = np.logical_and(np.logical_and(our_whites, their_blacks), random_numbers > self.epsilon)
        while np.any(to_be_replaced):
            them_idx[to_be_replaced] = np.random.choice(us_idx, size=np.count_nonzero(to_be_replaced), replace=True)
            their_blacks = population.races[them_idx] == model.black
            to_be_replaced = np.logical_and(np.logical_and(our_whites, their_blacks), random_numbers > self.epsilon)

        # (black, white) pair
        our_blacks = population.races[us_idx] == model.black
        their_whites = population.races[them_idx] == model.european
        random_numbers = np.random.random(size=our_blacks.shape)
        to_be_replaced = np.logical_and(np.logical_and(our_whites, their_blacks), random_numbers > self.epsilon)
        while np.any(to_be_replaced):
            them_idx[to_be_replaced] = np.random.choice(us_idx, size=np.count_nonzero(to_be_replaced), replace=True)
            their_whites = population.races[them_idx] == model.european
            to_be_replaced = np.logical_and(np.logical_and(our_blacks, their_whites), random_numbers > self.epsilon)

        return us_idx, them_idx

    @property
    def __name__(self):
        return f'{self.__class__.__name__}({self.epsilon})'
