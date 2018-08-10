import unittest
import os

TEST_PATH = './data/test/'


class TestAccuracy(unittest.TestCase):
    def test_can_get_label_names(self):
        labels = os.listdir(TEST_PATH)
        self.assertTrue('eight' in labels)

    def test_can_get_label_counts(self):
        sounds = os.listdir(TEST_PATH + '/bed')
        self.assertEqual(len(sounds), 395)

