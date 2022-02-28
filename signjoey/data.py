# coding: utf-8
"""
Data module
"""
import os
import sys
import random

import numpy as np
import torch
from torchtext import data
from torchtext.data import Dataset, Iterator
import socket

from signjoey import data_preprocessing
from signjoey.dataset import SignTranslationDataset
from signjoey.vocabulary import (
    build_vocab,
    Vocabulary,
    UNK_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
    PAD_TOKEN,
)


def load_data(data_cfg: dict, keep_only = None) -> (Dataset, Dataset, Dataset, Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    If you set ``random_dev_subset``, a random selection of this size is used
    from the dev development instead of the full development set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuration file)
    :param keep_only: If not `None`, only the samples with this name will be kept. A dict with keys 'train', 'dev', and 'test', each containing a list of sample names.
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: test dataset if given, otherwise None
        - gls_vocab: gloss vocabulary extracted from training data
        - txt_vocab: spoken text vocabulary extracted from training data
    """

    if keep_only is None:
        keep_only = {'train': None, 'dev': None, 'test': None}

    data_path = data_cfg.get("data_path", "./data")

    if isinstance(data_cfg["train"], list):
        # If we combine multiple features, we need all paths.
        train_paths = [os.path.join(data_path, x) for x in data_cfg["train"]]
        dev_paths = [os.path.join(data_path, x) for x in data_cfg["dev"]]
        test_paths = [os.path.join(data_path, x) for x in data_cfg["test"]]
        pad_feature_size = sum(data_cfg["feature_size"])
    else:
        # Otherwise, we only need the single path.
        train_paths = os.path.join(data_path, data_cfg["train"])
        dev_paths = os.path.join(data_path, data_cfg["dev"])
        test_paths = os.path.join(data_path, data_cfg["test"])
        pad_feature_size = data_cfg["feature_size"]

    level = data_cfg["level"]
    txt_lowercase = data_cfg["txt_lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    preprocessing = data_cfg.get("preprocessing", None)
    if preprocessing is None:
        preprocess_features = lambda tokens: tokens
    else:
        if preprocessing == "remove_keypoints_and_normalize":
            preprocess_features = data_preprocessing.remove_keypoints_and_normalize
        else:
            print(f'Unknown preprocessing function {preprocessing}, using identity function')
            preprocess_features = lambda tokens: tokens

    def tokenize_text(text):
        if level == "char":
            return list(text)
        else:
            return text.split()

    def tokenize_features(features):
        # Creates as many chunks as there are elements in the feature sequence.
        ft_list = torch.split(features, 1, dim=0)
        # Now, return a list of chunks (as a list of tokens).
        return [ft.squeeze() for ft in ft_list]

    def stack_features(features, _):
        # Makes sure that the features are all a single tensor.
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)

    sequence_field = data.RawField()
    signer_field = data.RawField()

    mBartVocab = 'mbart50' in data_cfg.get('train')
    print('Using mBART vocabulary?', mBartVocab)

    sgn_field = data.Field(
        use_vocab=False,
        init_token=None,
        eos_token=None,
        dtype=torch.float32,
        preprocessing=preprocess_features,  # After tokenizing but before numericalizing.
        tokenize=tokenize_features,  # Tokenization function.
        batch_first=True,
        include_lengths=True,
        postprocessing=stack_features,  # After numericalizing but before turning into a tensor.
        pad_token=torch.zeros((pad_feature_size,)),
    )

    gls_field = data.Field(
        pad_token=PAD_TOKEN,
        tokenize=tokenize_text,
        batch_first=True,
        lower=False,
        include_lengths=True,
    )

    txt_field = data.Field(
        init_token=BOS_TOKEN if not mBartVocab else None,
        # For mBartVocab, 'de_DE' (in data file) is start token for training, and EOS for prediction.
        eos_token=EOS_TOKEN if not mBartVocab else None,  # For mBartVocab, '</s>' is already included in the data file
        pad_token=PAD_TOKEN,
        tokenize=tokenize_text,
        unk_token=UNK_TOKEN,
        batch_first=True,
        lower=txt_lowercase,
        include_lengths=True,
    )

    txt_field_pred = data.Field(
        # init_token is only difference
        init_token=BOS_TOKEN if not mBartVocab else EOS_TOKEN,  # EOS for prediction (config.decoder_start_token_id)
        eos_token=EOS_TOKEN if not mBartVocab else None,  # For mBartVocab, '</s>' is already included in the data file
        pad_token=PAD_TOKEN,
        tokenize=tokenize_text,
        unk_token=UNK_TOKEN,
        batch_first=True,
        lower=txt_lowercase,
        include_lengths=True,
    )

    train_data = SignTranslationDataset(
        path=train_paths,
        fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
        filter_pred=lambda x: len(vars(x)["sgn"]) <= max_sent_length
                              and len(vars(x)["txt"]) <= max_sent_length,
        keep_only=keep_only['train']
    )

    gls_max_size = data_cfg.get("gls_voc_limit", sys.maxsize)
    gls_min_freq = data_cfg.get("gls_voc_min_freq", 1)
    txt_max_size = data_cfg.get("txt_voc_limit", sys.maxsize)
    txt_min_freq = data_cfg.get("txt_voc_min_freq", 1)

    gls_vocab_file = data_cfg.get("gls_vocab", None)
    txt_vocab_file = data_cfg.get("txt_vocab", None)

    gls_vocab = build_vocab(
        field="gls",
        min_freq=gls_min_freq,
        max_size=gls_max_size,
        dataset=train_data,
        vocab_file=gls_vocab_file,
    )

    txt_vocab = build_vocab(
        field="txt",
        min_freq=txt_min_freq,
        max_size=txt_max_size,
        dataset=train_data,
        vocab_file=txt_vocab_file,
        mBartVocab=mBartVocab,
    )
    random_train_subset = data_cfg.get("random_train_subset", -1)
    if random_train_subset > -1:
        # select this many training examples randomly and discard the rest
        keep_ratio = random_train_subset / len(train_data)
        keep, _ = train_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio], random_state=random.getstate()
        )
        train_data = keep
    learning_curve_percentage = data_cfg.get("learning_curve_percentage", 1.0)
    if not np.isclose(learning_curve_percentage, 1.0):
        keep, _ = train_data.split(split_ratio=[learning_curve_percentage, 1 - learning_curve_percentage],
                                   random_state=random.getstate())
        train_data = keep

    dev_data = SignTranslationDataset(
        path=dev_paths,
        fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field_pred),
        keep_only=keep_only['dev']
    )
    random_dev_subset = data_cfg.get("random_dev_subset", -1)
    if random_dev_subset > -1:
        # select this many development examples randomly and discard the rest
        keep_ratio = random_dev_subset / len(dev_data)
        keep, _ = dev_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio], random_state=random.getstate()
        )
        dev_data = keep

    # check if target exists
    test_data = SignTranslationDataset(
        path=test_paths,
        fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field_pred),
        keep_only=keep_only['test']
    )

    gls_field.vocab = gls_vocab
    txt_field.vocab = txt_vocab
    txt_field_pred.vocab = txt_vocab
    return train_data, dev_data, test_data, gls_vocab, txt_vocab


# TODO (Cihan): I don't like this use of globals.
#  Need to find a more elegant solution for this it at some point.
# pylint: disable=global-at-module-level
global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)"""
    global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch
    if count == 1:
        max_sgn_in_batch = 0
        max_gls_in_batch = 0
        max_txt_in_batch = 0
    max_sgn_in_batch = max(max_sgn_in_batch, len(new.sgn))
    max_gls_in_batch = max(max_gls_in_batch, len(new.gls))
    max_txt_in_batch = max(max_txt_in_batch, len(new.txt) + 2)
    sgn_elements = count * max_sgn_in_batch
    gls_elements = count * max_gls_in_batch
    txt_elements = count * max_txt_in_batch
    return max(sgn_elements, gls_elements, txt_elements)


def make_data_iter(
        dataset: Dataset,
        batch_size: int,
        batch_type: str = "sentence",
        train: bool = False,
        shuffle: bool = False,
) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing sgn and optionally txt
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False,
            sort=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=True,
            sort_within_batch=True,
            sort_key=lambda x: len(x.sgn),
            shuffle=shuffle,
        )
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=False,
            sort=False,
        )

    return data_iter
