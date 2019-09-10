import glob
import tensorflow as tf
import pickle
import numpy as np
from random import randint

from data import dataset


class Rand10(dataset.Dataset):

    def __init__(self, opt):
        super(Rand10, self).__init__(opt)

        self.num_threads = 8

        self.list_labels = range(2)
        self.num_images_training = 100000
        self.num_images_test = 100000

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

        train = self.__unpickle('train_10.pickle')
        val = self.__unpickle('val_10.pickle')

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
        test = self.__unpickle('test_10.pickle')

        test_addrs = []
        for t in test['data']:
            test_addrs.append(t)
        test_labels = test['labels'].tolist()

        return test_addrs, test_labels

    def preprocess_image(self, augmentation, standarization, image):
        return image
