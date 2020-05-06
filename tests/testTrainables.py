import unittest
from data.dataset import Dataset, RawTorchRecording, Thinker

from .testDataset import create_dummy_raw


class TestSimpleClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset

