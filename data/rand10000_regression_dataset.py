import glob
import tensorflow as tf
import pickle
import numpy as np
from random import randint

from data import dataset


class Rand10000_regression(dataset.Dataset):

    def __init__(self, opt):
        super(Rand10000_regression, self).__init__(opt, opt.hyper.mse)

        self.num_threads = 8

        self.list_labels = None
        self.num_images_training = 50000
        self.num_images_test = 50000

        self.num_images_epoch = self.num_images_training
        self.num_images_val = self.num_images_training

        self.create_tfrecords()

    # Helper functions:
    def __unpickle(self, file_name):
        with open(file_name, 'rb') as fo:
            data = pickle.load(fo)
        return data

    # Virtual functions:
    def get_data_trainval(self):

        train = self.__unpickle('/om/user/scasper/data/synthetic_regression/train_10000.pickle')
        val = self.__unpickle('/om/user/scasper/data/synthetic_regression/val_10000.pickle')

        train_addrs = []
        for t in train['data']:
            train_addrs.append(t)
        train_labels = train['labels'].tolist()

        val_addrs = []
        for t in val['data']:
            val_addrs.append(t)
        val_labels = val['labels'].tolist()

        return train_addrs, train_labels, val_addrs, val_labels

    def get_data_test(self):
        test = self.__unpickle('/om/user/scasper/data/synthetic_regression/test_10000.pickle')

        test_addrs = []
        for t in test['data']:
            test_addrs.append(t)
        test_labels = test['labels'].tolist()

        return test_addrs, test_labels

    def preprocess_image(self, augmentation, standarization, image):
        return image
