import datetime
import itertools
import time

import numpy as np

from pydgin import multiprocess, selections, serialization


class MultiRunner:
    """ Runs the simulation against a full feature set with every possible combination of given inputs.

    Estimates remaining time along the way.

    Use `MultiRunner.from_paths` to get an instance with the appropriate data then use `simulate_all_combinations` to
    run it with given parameters.
    """

    @classmethod
    def from_paths(cls, population_deltas_path, epochs_path, all_features_path, **kwargs):
        with open(population_deltas_path, 'r') as population_deltas_fd, \
                open(epochs_path, 'r') as epochs_fd, \
                open(all_features_path, 'r') as all_features_fd:
            return cls.from_files(population_deltas_fd, epochs_fd, all_features_fd, **kwargs)

    @classmethod
    def from_files(cls, population_deltas_fd, epochs_fd, all_features_fd, **kwargs):
        return cls(serialization.parse_matrix(population_deltas_fd),
                   serialization.parse_vector(epochs_fd),
                   serialization.parse_matrix(all_features_fd),
                   **kwargs)

    def __init__(self, population_deltas, epochs, all_features, log_func=None, n_processes=6):
        self.population_deltas = population_deltas
        self.epochs = epochs
        self.all_features = all_features
        self.mauritian_creole = all_features[:, 0]
        self.french = all_features[:, 1]
        self.log = log_func or print
        self.n_processes = n_processes

    def _distribution_one(self, lc, lu, non_learning_europeans, select_func, plantation_epoch):
        distributions_promise = multiprocess.run_all_features(
            self.all_features,
            self.population_deltas,
            self.epochs,
            lc, lu, select_func, non_learning_europeans,
            plantation_epoch,
            n_processes=self.n_processes,
        )
        return distributions_promise.wait()

    def _initial(self, non_learning_europeans, selection_functions, plantation_epochs, lcs, lu_diffs):
        self.tooks = []
        self.remaining_iterations = len(list(itertools.product(
            non_learning_europeans, selection_functions, plantation_epochs, lcs, lu_diffs,
        )))

    def _pre_run(self, lc, lu, non_learning_europeans, select_func):
        self._start_time = time.time()

        header = (f'Learning: {lc:8.2f} | {lu:8.3f}    '
                  f'Europeans learn: {"no " if non_learning_europeans else "yes"}    '
                  f'Selection function: {select_func.__name__}')

        # Log some stuff
        self.log('=' * len(header))
        self.log(header)
        self.log('-' * len(header))
        self.log(f'{timestr()}\tRunning simulation...')

    def _post_run(self):
        end_time = time.time()
        self.tooks.append(end_time - self._start_time)
        del self._start_time
        self.remaining_iterations -= 1

        eta = self.remaining_iterations * np.mean(self.tooks) * 1.5  # 1.5 seems to make it more accurate for now
        remaining_string = f'{eta // 3600} hours and {eta // 60 % 60} minutes'

        # Log time
        self.log(
            f'{timestr()}\tDone simulating, took {self.tooks[-1]:.2f}s, '
            f'{self.remaining_iterations} runs remain, estimate finished in {remaining_string}'
        )

    def simulate_all_combinations(self, *,
                                  lcs, lu_diffs, non_learning_europeanses, selection_functions, plantation_epochs):
        outputs = {}

        for non_learning_europeans, select_func, plantation_epoch in itertools.product(
                non_learning_europeanses, selection_functions,
                plantation_epochs,
        ):
            self._initial(non_learning_europeanses, selection_functions, plantation_epochs, lcs, lu_diffs)

            predictions = {}
            for lc in lcs:
                for lu_d in lu_diffs:
                    lu = lc - lu_d
                    self._pre_run(lc, lu, non_learning_europeans, select_func)

                    # Actually calculate the thing
                    distributions = self._distribution_one(
                        lc, lu, non_learning_europeans, select_func, plantation_epoch,
                    )

                    self._post_run()

                    # Put it in the output
                    predicted_language = [dist.winning_feature_index for dist in distributions]
                    predictions.setdefault(lc, {})[lu] = predicted_language

                outputs[(non_learning_europeans, select_func.__name__, plantation_epoch)] = predictions
        return outputs


def timestr():
    return datetime.datetime.now().strftime('%c')


default_all = {
    'lcs': [0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5],
    'lu_diffs': np.arange(0.003, 0.01, 0.001),
    'non_learning_europeans': [True, False],
    'selections': [selections.random, selections.Segregation(0.1), selections.ColdShoulder(0.1)],
    'plantation_epochs': [2874, 3058, 3122, 3213, None],
}


def _get_selection(select_str: str):
    import importlib
    module, expression = select_str.rsplit(':', maxsplit=1)
    mod = importlib.import_module(module)
    return eval(f'mod.{expression}', {'mod': mod})


def main():
    import argparse
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--demography', default=os.path.join(root, 'inputs', 'Demography.txt'))
    ap.add_argument('-t', '--time', default=os.path.join(root, 'inputs', 'Time.txt'))
    ap.add_argument('-f', '--features', default=os.path.join(root, 'inputs', 'useful_features_with_mc.txt'))
    ap.add_argument('--lcs', nargs='+', type=float, required=True)
    ap.add_argument('--lu-diffs', nargs='+', type=float, required=True)
    ap.add_argument('--nle', nargs='+', type=bool, default=[False])
    ap.add_argument('--select', nargs='+', default=['pydgin.selections:Segregation(0.1)'])
    ap.add_argument('--plantation-epochs', nargs='+', default=[0], type=int)
    ap.add_argument('output', nargs='?', default='pydgin_outputs/db.pkl')

    args = ap.parse_args()

    runner = MultiRunner.from_paths(args.demography, args.time, args.features)

    outputs = runner.simulate_all_combinations(
        lcs=args.lcs,
        lu_diffs=args.lu_diffs,
        non_learning_europeanses=args.nle,
        selection_functions=[_get_selection(selection) for selection in args.select],
        plantation_epochs=args.plantation_epochs,
    )
    serialization.save_many_run(outputs, args.output)


if __name__ == '__main__':
    main()
