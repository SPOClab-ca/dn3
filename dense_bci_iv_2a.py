import mne
import copy
import argparse

from tensorflow import keras
from keras_radam.training import RAdamOptimizer

from models import DenseTCNN, ShallowConvNet, ShallowFBCSP, DenseSCNN, SCNN
from datasets import BNCI2014001, PhysioNetMMI
from dataloaders import EpochsDataLoader, labelled_dataset_concat
from preprocess import EApreprocessor
from utils import dataset_concat, CosineScheduleLR
from transforms import LabelSmoothing, Mixup, OneHotLabels

DATASET = BNCI2014001()
# DATASET = PhysioNetMMI()


def single_subject(subject, epoched):
    training = dataset_concat(*[s.train_dataset() for s in epoched[subject]['session_T'].values()])
    testing = dataset_concat(*[s.eval_dataset() for s in epoched[subject]['session_E'].values()])
    return training, testing


def all_subjects_split(subject, validation_subject, epoched: dict):
    epoched = copy.copy(epoched)
    test_subject = epoched.pop(subject)
    validation_subject = epoched.pop(validation_subject)

    test = dataset_concat(*[s.eval_dataset() for s in test_subject['session_E'].values()])
    validation = dataset_concat(*[s.eval_dataset() for s in validation_subject['session_E'].values()])
    training = dataset_concat(
        # All T and E sessions for all remaining subjects
        dataset_concat(*[epoched[b][s][r].train_dataset() for b in epoched for s in epoched[b] for r in epoched[b][s]]),
        # Duplicated T sessions for the validation and test subjects (to balance size of domains)
        dataset_concat(*[s.train_dataset() for s in validation_subject['session_T'].values()]).repeat(2),
        dataset_concat(*[s.train_dataset() for s in test_subject['session_T'].values()]).repeat(2)
    )
    return training, validation, test


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
                                                                      picks=['eeg', 'eog'],)
                                                                      # preprocessings=[EApreprocessor()])
    return loaders


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example for using a meta-model and reptile meta-optimization to "
                                                 "leverage Thinker Invariance and few-shot augmentation.")
    parser.add_argument('--tmin', default=-0.5, type=float, help='Start time for epoching')
    parser.add_argument('--tlen', default=4.5, type=float, help='Length per epoch.')
    args = parser.parse_args()

    mne.set_log_level(False)

    loaders = data_as_loaders(args)

    for i, subject in enumerate(DATASET.subjects()):
        training, validation, test = all_subjects_split(subject, DATASET.subjects()[i - 1], loaders)
        training = training.shuffle(6000).batch(16, drop_remainder=True)

        validation = validation.batch(32)
        test = test.batch(32)

        # model = DenseTCNN(targets=4, channels=25, samples_t=int(250*args.tlen))
        # model = ShallowConvNet(4, Chans=25, Samples=int(250*args.tlen))
        model = DenseSCNN(targets=4, channels=25, samples=int(250 * args.tlen))
        # model = SCNN(targets=4, channels=25, samples=int(250 * args.tlen))
        model.summary()
        model.compile(optimizer=RAdamOptimizer(learning_rate=1e-3, weight_decay=1e-2),
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'],)

        model.fit(x=training, validation_data=validation, epochs=100,)
                  # callbacks=[keras.callbacks.LearningRateScheduler(lambda e, r: r if e % 10 else e / 10)])

        model.evaluate(test)

        break
