# Copyright 2019 SanghunYun, Korea University.
# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)
# 
# SanghunYun, Korea University refered Dong-Hyun Lee, Kakao Brain's code (class model)
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


import json
from typing import NamedTuple


class params(NamedTuple):

    ############################ guide #############################
    # lr(learning rate) : fine_tune(2e-5), futher-train(1.5e-4~2e-5)
    # mode : train, eval, test
    # uda_mode : True, False
    # total_steps : n_epochs * n_examples / 3
    # max_seq_length : 128, 256, 512
    # unsup_ratio : more than 3
    # uda_softmax_temp : more than 0.5
    # uda_confidence_temp : ??
    # tsa : linear_schedule
    ################################################################

    # train
    seed: int = 1421
    lr: int = 2e-5                      # lr_scheduled = lr * factor
    # n_epochs: int = 3
    warmup: float = 0.1                 # warmup steps = total_steps * warmup
    do_lower_case: bool = True
    mode: str = None                    # train, eval, test
    uda_mode: bool = False              # True, False
    
    total_steps: int = 100000           # total_steps >= n_epcohs * n_examples / 3
    max_seq_length: int = 128
    train_batch_size: int = 32
    eval_batch_size: int = 8

    # unsup
    unsup_ratio: int = 0                # unsup_batch_size = unsup_ratio * sup_batch_size
    uda_coeff: int = 1                  # total_loss = sup_loss + uda_coeff*unsup_loss
    tsa: str = 'linear_schedule'           # log, linear, exp
    uda_softmax_temp: float = -1        # 0 ~ 1
    uda_confidence_thresh: float = -1   # 0 ~ 1

    # data
    data_parallel: bool = True
    need_prepro: bool = False           # is data already preprocessed?
    sup_data_dir: str = None
    unsup_data_dir: str = None
    eval_data_dir: str = None
    n_sup: int = None
    n_unsup: int = None

    model_file: str = None              # fine-tuned
    pretrain_file: str = None           # pre-trained
    vocab: str = None
    task: str = None

    # results
    save_steps: int = 100
    check_steps: int = 10
    results_dir: str = None

    # appendix
    is_position: bool = False           # appendix not used
    
    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, 'r')))


class pretrain(NamedTuple):
    seed: int = 3232
    batch_size: int = 32
    lr: int = 1.5e-4
    n_epochs: int = 100
    warmup: float = 0.1
    save_steps: int = 100
    total_steps: int = 100000
    results_dir : str = None
    
    # do not change
    uda_mode: bool = False

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, 'r')))



class model(NamedTuple):
    "Configuration for BERT model"
    vocab_size: int = None # Size of Vocabulary
    dim: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12 # Numher of Hidden Layers
    n_heads: int = 12 # Numher of Heads in Multi-Headed Attention Layers
    dim_ff: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1 # Probability of Dropout of Attention Layers
    max_len: int = 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, 'r')))