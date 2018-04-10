#!/usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing

import ipywidgets as widgets

from pydgin import model, selections, serialization


class ModelRunner(object):
    def __init__(self, *, model):
        self.model = model

    def __call__(self, args):
        i, feature_columns = args
        result = self.model.run(feature_columns)
        return i, result


class Promise(object):
    def __init__(self, pool, generator, count):
        self.generator = generator
        self.progress = widgets.IntProgress(max=count)
        self.pool = pool

    def wait(self):
        if hasattr(self, 'results'):
            return self.results
        self.results = [m for i, m in sorted(list(self._yield_wait()))]
        self.pool.close()
        self.pool.join()
        return self.results

    def _yield_wait(self):
        for model in self.generator:
            yield model
            self.progress.value += 1


def run_all_features(all_features, population_deltas, epochs, learning_rate_coordinated,
                     learning_rate_uncoordinated, selection_function, non_learning_europeans, plantation_epoch,
                     n_processes=6):
    """ Run the model on every given feature with the supplied arguments in parallel. """
    pool = multiprocessing.Pool(processes=n_processes)
    m = model.Model(
        population_deltas=population_deltas,
        epoch_lengths=epochs,
        select=selection_function,
        lr_coordinated=learning_rate_coordinated,
        lr_uncoordinated=learning_rate_uncoordinated,
        non_learning_europeans=non_learning_europeans,
        plantation_epoch=plantation_epoch,
    )
    run_model = ModelRunner(model=m)
    results = pool.imap_unordered(run_model, enumerate(all_features))
    return Promise(pool, results, all_features.shape[0])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('demography')
    ap.add_argument('features')
    ap.add_argument('epochs')
    ap.add_argument('-c', '--learning-coordinated', type=float)
    ap.add_argument('-u', '--learning-uncoordinated', type=float)
    ap.add_argument('--non-learning.europeans', action='store_true')
    ap.add_argument('-j', '--n-processes', type=int, default=6)
    ap.add_argument('--plantation-epoch', type=int, default=None)
    ap.add_argument('-e', '--epsilon', type=float, default=0.1)
    args = ap.parse_args()

    with open(args.demography) as demography:
        population_deltas = serialization.parse_matrix(demography)

    with open(args.features) as features:
        all_features = serialization.parse_matrix(features)

    with open(args.epochs) as epochs_file:
        epochs = serialization.parse_vector(epochs_file)

    distributions = run_all_features(all_features, population_deltas, epochs, args.learning_coordinated,
                                     args.learning_uncoordinated, selections.ColdShoulder(args.epsilon),
                                     args.non_learning_europeans, args.plantation_epoch,
                                     args.n_processes)

    for d in distributions.wait():
        print(f'{d.mean_features}, {d.winning_feature_index}')


if __name__ == '__main__':
    main()
