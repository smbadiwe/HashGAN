# =============================================================================
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import math
import cv2
import numpy as np
import tensorflow as tf


class Dataset(object):
    def __init__(self, list_path, image_root, train=True, height_width=256):
        if list_path:
            self.lines = open(list_path, 'r').readlines()
        else:
            self.lines = []
        self.image_root = image_root
        self.n_samples = len(self.lines)
        self.train = train
        self.height_width = height_width
        self.img_shape = (self.height_width, self.height_width)

        self._img = [0] * self.n_samples
        self._label = [0] * self.n_samples
        self._filenames = [None] * self.n_samples
        self._load = [0] * self.n_samples
        self._load_num = 0
        self._status = 0
        self.data = self.img_data

    def get_file_path_of_image_at(self, index):
        filename = self.lines[index].strip().split()[0]
        return os.path.join(self.image_root, filename)

    def read_image_at(self, index):
        path = self.get_file_path_of_image_at(index)
        img = cv2.imread(path)
        return cv2.resize(img, self.img_shape, interpolation=cv2.INTER_AREA)

    def read_image(self, filename):
        path = os.path.join(self.image_root, filename)
        img = cv2.imread(path)
        return cv2.resize(img, self.img_shape, interpolation=cv2.INTER_AREA)

    def get_label(self, index):
        return [int(j) for j in self.lines[index].strip().split()[1:]]

    def img_data(self, index):
        if self._status:
            return self._img[index, :], self._label[index, :], self._filenames[index]
        else:
            ret_img = []
            ret_label = []
            ret_filenames = []
            for i in index:
                # noinspection PyBroadException,PyPep8
                try:
                    filename = self.lines[i].strip().split()[0]
                    if self.train:
                        if not self._load[i]:
                            self._img[i] = self.read_image(filename)
                            self._label[i] = self.get_label(i)
                            self._filenames[i] = filename
                            self._load[i] = 1
                            self._load_num += 1
                        ret_img.append(self._img[i])
                        ret_label.append(self._label[i])
                    else:
                        self._label[i] = self.get_label(i)
                        ret_img.append(self.read_image(filename))
                        ret_label.append(self._label[i])
                    ret_filenames.append(filename)
                except:
                    print('cannot open', self.lines[i])
                    raise

            if self._load_num == self.n_samples:
                self._status = 1
                self._img = np.asarray(self._img)
                self._label = np.asarray(self._label)
                self._filenames = np.asarray(self._filenames)
            return np.asarray(ret_img), np.asarray(ret_label), np.asarray(ret_filenames)


class Dataloader(object):

    def __init__(self, batch_size, width_height, list_root, image_root):
        self.batch_size = batch_size
        self.width_height = width_height
        self.data_root = list_root
        self.image_root = image_root

    def data_generator(self, split):
        _dataset = Dataset(list_path=os.path.join(self.data_root, split + '.txt'),
                           image_root=self.image_root, train=True, height_width=self.width_height)

        def get_epoch():
            with tf.device('/cpu:0'):
                _index_in_epoch = 0
                _perm = np.arange(_dataset.n_samples)
                np.random.shuffle(_perm)
                for _ in range(int(math.ceil(_dataset.n_samples / self.batch_size))):
                    start = _index_in_epoch
                    _index_in_epoch += self.batch_size
                    # finish one epoch
                    if _index_in_epoch > _dataset.n_samples:
                        data, label, data_paths = _dataset.data(_perm[start:])
                        data1, label1, data_paths1 = _dataset.data(_perm[:_index_in_epoch - _dataset.n_samples])
                        data = np.concatenate([data, data1], axis=0)
                        label = np.concatenate([label, label1], axis=0)
                        data_paths = np.concatenate([data_paths, data_paths1], axis=0)
                    else:
                        end = _index_in_epoch
                        data, label, data_paths = _dataset.data(_perm[start:end])

                    # n*h*w*c -> n*c*h*w
                    data = np.transpose(data, (0, 3, 1, 2))
                    # bgr -> rgb
                    data = data[:, ::-1, :, :]
                    data = np.reshape(data, (self.batch_size, -1))
                    yield (data, label, data_paths)

        return get_epoch

    @property
    def train_gen(self):
        return self.data_generator('train')

    @property
    def test_gen(self):
        return self.data_generator('test')

    @property
    def db_gen(self):
        return self.data_generator('database')

    @property
    def unlabeled_db_gen(self):
        return self.data_generator('database_nolabel')

    @staticmethod
    def inf_gen(gen):
        def generator():
            while True:
                for images_iter_, labels_iter_ in gen():
                    return images_iter_, labels_iter_
        return generator
