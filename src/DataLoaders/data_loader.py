from itertools import count

import numpy as np
from typing import Generator, Tuple, Literal


class DataLoader:
    def __init__(self, features: np.ndarray, labels: np.ndarray, train_ratio, valid_ratio, test_ratio, seed=420):

        denom = train_ratio + valid_ratio + test_ratio
        num_train = int(train_ratio / denom * len(features))
        num_valid = int(valid_ratio / denom * len(features))
        num_test = int(test_ratio / denom * len(features))

        ### Check invariants

        if features is None or labels is None:
            raise ValueError("Features and labels cannot be None")
        if np.shape(features)[0] != np.shape(labels)[0]:
            raise ValueError("Features and labels must have the same number of samples")
        if num_train < 0 or num_valid < 0 or num_test < 0:
            raise ValueError("Number of train, valid, and test samples must be non-negative")
        if num_train + num_valid + num_test > len(features):
            raise ValueError("Sum of train, valid, and test samples must not exceed total number of samples")

        # 
        self.features = features
        self.labels = labels

        self.num_train = num_train
        self.num_valid = num_valid
        self.num_test = num_test

        self.seed = seed

        ### Obtain indices

        # Shuffle for training data
        np.random.seed(self.seed)
        indices = np.random.permutation(num_train)

        self.train_indices = indices[:self.num_train]
        self.valid_indices = np.arange(self.num_train, self.num_train + self.num_valid)
        self.test_indices = np.arange(self.num_train + self.num_valid, self.num_train + self.num_valid + self.num_test)

    def zip_generators(self, train_generator, validation_generator):
        """
        Utility function to zip one generator with another. Zip will have same length as train_generator.
        Two cases: (1) len train <= len validate and (2) len train > len validate.
            (1) batches yielded from both train and validate; no None
            (2) batches yielded from train; validate yields None for |train| - |validate| batches, then validate starts yielding

        :param train_generator:
        :param validation_generator:
        :return tuple: (batch_train_x, batch_train_y), (batch_validate_x, batch_validate_y), where validate_x and validate_y may be None
        """

        prev_yields = count(0)
        if next(prev_yields) < self.num_train - self.num_valid:
            yield next(train_generator), (None, None)
        yield from zip(train_generator, validation_generator)


    def batch_generator(self, batch_size=32, mode: Literal["train", "validate", "test"] = "train") -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generator that yields batches of train_x and train_y.

        :param batch_size: (int) The size of each batch.
        :param mode: (Literal["train", "validate", "test"]) Specifies subset of data to yield.
        :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
        """

        # Select the appropriate indices based on the mode.
        if mode == "train":
            indices = self.train_indices
            total_samples = self.num_train
        elif mode == "validate":
            indices = self.valid_indices
            total_samples = self.num_valid
        elif mode == "test":
            indices = self.test_indices
            total_samples = self.num_test
        else:
            raise ValueError("Invalid mode. Choose from 'train', 'validate', or 'test'.")

        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_indices = indices[start_idx:end_idx]
            batch_x = self.features[batch_indices]
            batch_y = self.labels[batch_indices]

            yield batch_x, batch_y