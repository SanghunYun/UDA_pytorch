# Copyright 2019 SanghunYun, Korea University.
# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# 
# This file has been modified by SanghunYun, Korea University
# for add fucntion of _get_device and class of output_logging.
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


import os
import csv
import random
import logging

import numpy as np
import torch


def torch_device_one():
    return torch.tensor(1.).to(_get_device())

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    "get device (CPU or GPU)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device

def _get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def get_random_word(vocab_words):
    i = random.randint(0, len(vocab_words)-1)
    return vocab_words[i]

def get_logger(name, log_path):
    "get logger"
    logger = logging.getLogger(name)
    fomatter = logging.Formatter(
        '[ %(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

    if not os.path.isfile(log_path):
        f = open(log_path, "w+")

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(fomatter)
    logger.addHandler(fileHandler)

    #streamHandler = logging.StreamHandler()
    #streamHandler.setFormatter(fomatter)
    #logger.addHandler(streamHandler)

    logger.setLevel(logging.DEBUG)
    return logger


class output_logging(object):
    def __init__(self, mode, real_time=False, dump_dir=None):
        self.mode = mode
        self.real_time = real_time
        self.dump_dir = dump_dir if dump_dir else None

        if dump_dir:
            self.dump = open(os.path.join(dump_dir, 'logs/output.tsv'), 'w', encoding='utf-8', newline='')
            self.wr = csv.writer(self.dump, delimiter='\t')

            # header
            if mode == 'eval':
                self.wr.writerow(['Ground_truth', 'Predcit', 'sentence'])
            elif mode == 'test':
                self.wr.writerow(['Predict', 'sentence'])

    def __del__(self):
        if self.dump_dir:
            self.dump.close()

    def logs(self, sentence, pred, ground_turth=None):
        if self.real_time:
            if self.mode == 'eval':
                for p, g, s in zip(pred, ground_turth, sentence):
                    print('Ground_truth | Predict')
                    print(int(g), '         ', int(p))
                    print(s, end='\n\n')
            elif self.mode == 'test':
                for p, s in zip(pred, sentence):
                    print('predict : ', int(p))
                    print(s, end='\n\n')
        
        if self.dump_dir:
            if self.mode == 'eval':
                for p, g, s in zip(pred, ground_turth, sentence):
                    self.wr.writerow([int(p), int(g), s])
            elif self.mode == 'test':
                for p, s in zip(pred, sentence):
                    self.wr.writerow([int(p), s])
