import mne
import copy
import argparse
import tqdm

from models import DenseTCNN, ShallowConvNet, ShallowFBCSP, DenseSCNN
from datasets import BNCI2014001
from dataloaders import EpochsDataLoader, labelled_dataset_concat
from metaopt import Reptile
from utils import dataset_concat
from tensorflow import keras
import tensorflow as tf

DATASET = BNCI2014001()


def loader_split(subject, validation_subject, epoched: dict):
    epoched = copy.copy(epoched)
    test_subject = epoched[subject].pop('session_E')
    validation_subject = epoched[validation_subject].pop('session_E')

    for subj in epoched:
        epoched[subj] = dataset_concat(
            *[run.train_dataset() for sess in epoched[subj] for run in epoched[subj][sess].values()]
        )
        if subj == subject or subj == validation_subject:
            epoched[subj] = epoched[subj].repeat(2)

    test = dataset_concat(*[s.eval_dataset() for s in test_subject.values()])
    validation = dataset_concat(*[s.eval_dataset() for s in validation_subject.values()])
    return epoched, validation, test


def data_as_loaders(args):
    loaders = dict()
    for subject in DATASET.raw_data:
        loaders[subject] = dict()
        for session in DATASET.raw_data[subject]:
            loaders[subject][session] = dict()
            for run in DATASET.raw_data[subject][session]:
                raw = DATASET.raw_data[subject][session][run]
                events = mne.find_events(raw)
                if events.shape[0] != 0:
                    loaders[subject][session][run] = EpochsDataLoader(raw, events, tmin=args.tmin, tlen=args.tlen,
                                                                      picks=['eeg', 'eog'])
    return loaders


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example for using a meta-model and reptile meta-optimization to "
                                                 "leverage Thinker Invariance and few-shot augmentation.")
    parser.add_argument('--tmin', default=-0.5, type=float, help='Start time for epoching')
    parser.add_argument('--tlen', default=4.5, type=float, help='Length per epoch.')
    parser.add_argument('--epochs', '-e', default=120, type=int)
    parser.add_argument('--label-smoothing', '-ls', default=0, type=float, help='A parameter from 0-1 to indicate the '
                                                                                'degree of label smoothing. (0:none)')
    args = parser.parse_args()

    mne.set_log_level(False)

    loaders = data_as_loaders(args)

    for i, subject in enumerate(DATASET.subjects()):
        training, validation, test = loader_split(subject, DATASET.subjects()[i - 1], loaders)
        validation = validation.batch(32)

        # model = DenseTCNN(targets=4, channels=25, samples_t=int(250*args.tlen))
        model = DenseSCNN(4, channels=25, samples=int(250 * args.tlen))
        # model = ShallowConvNet(4, Chans=25, Samples=int(250*args.tlen))
        model.summary()
        optimizer = keras.optimizers.Adam(1e-3, amsgrad=False, beta_1=0)
        model.compile(optimizer=optimizer,
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        metaopt = Reptile()

        # Train
        training_loop = tqdm.trange(args.epochs, unit='epochs')
        for e in training_loop:
            metaopt.train(training, model, batch_sizes=32, outer_lr=1.0, inner_iterations=10)
            if e % 2 == 0:
                metrics = model.evaluate(validation)
                training_loop.set_postfix(lr=optimizer.lr.numpy())

        # Test
        # Before few-shot
        model.evaluate(test)
        # Few-shot
        metaopt.few_shot_evaluation(test, model, num_targets=4, num_shots=5)



