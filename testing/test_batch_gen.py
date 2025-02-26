import sys
import os
import unittest
import numpy as np
# Changed import to prepend src.
from src.DataLoaders.data_loader import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

class TestBatchGenerator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        """
        empty = np.array([])
        array_1 = np.array([[1]])
        array_3 = np.array([[1], [2], [3]])

        array_10 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [1, -4], [0, 3], [1, 9], [0, 0], [-5, -4], [8, -4]])
        array_10_sums = np.array([[3], [7], [11], [15], [-3], [3], [10], [0], [-9], [4]])

        self.dataset_empty = (empty, empty)
        self.dataset_1 = (array_1, array_1)
        self.dataset_3 = (array_3, array_3)
        self.dataset_basic_tuple_10 = (array_10, array_10_sums)

        self.batch_basic_numpy_tuple_even_divides = 5
        self.batch_basic_tuple_unevenly_divides = 3

    def test_generator_invariants(self):
        """
        """

        # Test seed not empty (nice for assignment)

        ### Dataset invariants (negative cases)

        one_tuple = (np.array([1]), )
        three_tuple = (np.array([1]), np.array([2]), np.array([3]))

        # not tuple
        with self.assertRaises(TypeError):
            data_loader = DataLoader("not a tuple", 10, 10, 10, 2)

        # not tuple of two elems
        with self.assertRaises(TypeError):
            data_loader = DataLoader(one_tuple, 0, 0, 0)

        with self.assertRaises(TypeError):
            data_loader = DataLoader(three_tuple, 0, 0, 0)

        # not tuple of equal elems
        left_heavy = (np.array([[1], [2]]), np.array([[0]]))
        right_heavy = (np.array([[0]]), np.array([[1], [2]]))

        assert(np.shape(left_heavy[0]) == (2, 1) and np.shape(left_heavy[1]) == (1, 1))

        with self.assertRaises(ValueError):
            data_loader = DataLoader(left_heavy, 0, 0, 0) 

        with self.assertRaises(ValueError):
            data_loader = DataLoader(right_heavy, 0, 0, 0) 
        
        # bad test splits

        with self.assertRaises(ValueError):
            data_loader = DataLoader(self.dataset_basic_tuple_10, -1, 0, 0) 

        with self.assertRaises(ValueError):
            data_loader = DataLoader(self.dataset_basic_tuple_10, 0, -1, 0) 
        
        with self.assertRaises(ValueError):
            data_loader = DataLoader(self.dataset_basic_tuple_10, 0, 0, -1) 

        with self.assertRaises(ValueError):
            data_loader = DataLoader(self.dataset_basic_tuple_10, 5, 5, 1)

    def test_batch_generator(self):
        num_data = len(self.dataset_basic_tuple_10[0])

        generator_10_uneven = DataLoader(self.dataset_basic_tuple_10, num_data, 0, 0).batch_generator(batch_size=4)
        batches_10_uneven = list(generator_10_uneven)

        generator_10_even = DataLoader(self.dataset_basic_tuple_10, num_data, 0, 0).batch_generator(batch_size=5)
        batches_10_even = list(generator_10_even)

        generator_1 = DataLoader(self.dataset_1, len(self.dataset_1[0]), 0, 0).batch_generator(batch_size=1)
        batches_1 = list(generator_1)

        num_data_3 = len(self.dataset_3[0])
        generator_3 = DataLoader(self.dataset_3, num_data_3, 0, 0).batch_generator(batch_size=(num_data_3 + 1))
        batches_3 = list(generator_3)

        # 10 samples tuple & batch divides unevenly
        self.assertEqual(batches_10_uneven[0][0].shape, (4, 2))
        self.assertEqual(batches_10_uneven[0][1].shape, (4, 1))
        
        self.assertEqual(batches_10_uneven[1][0].shape, (4, 2))
        self.assertEqual(batches_10_uneven[1][1].shape, (4, 1))

        self.assertEqual(batches_10_uneven[2][0].shape, (2, 2))
        self.assertEqual(batches_10_uneven[2][1].shape, (2, 1))

        self.assertEqual(len(batches_10_uneven), 3)

        # 10 samples numpy tuple & batch divides evenly
        self.assertEqual(len(batches_10_even), 2)

        self.assertEqual(batches_10_even[0][0].shape, (5, 2))
        self.assertEqual(batches_10_even[0][1].shape, (5, 1))

        self.assertEqual(batches_10_even[1][0].shape, (5, 2))
        self.assertEqual(batches_10_even[1][1].shape, (5, 1))

        # one data, batch of one
        self.assertEqual(len(batches_1), 1)
        self.assertEqual(batches_1[0][0].shape, (1, 1))
        self.assertEqual(batches_1[0][1].shape, (1, 1))

        # first batch overshoots
        self.assertEqual(len(batches_3), 1)
        self.assertEqual(batches_3[0][0].shape, (3, 1))
        self.assertEqual(batches_3[0][1].shape, (3, 1))

        # Bad batch size
        with self.assertRaises(ValueError):
            bad_generator = DataLoader(self.dataset_1, 1, 0, 0).batch_generator(batch_size=0)
            bad_batches = list(bad_generator)


    def try_test_seed(self, dataset, batch_size, seed):
        """
        """

        # same seed gives same training shuffle

        num_data = len(dataset[0])

        good_loader = DataLoader(dataset, num_data, 0, 0, seed) 
        same_loader = DataLoader(dataset, num_data, 0, 0, seed)

        first_gen = good_loader.batch_generator(batch_size=batch_size)
        second_gen = same_loader.batch_generator(batch_size=batch_size)

        # lists of tuples of numpy arrays - our test suite can't compare those directly
        # ... so we translate to normal python lists
        first_batches = list(first_gen)
        second_batches = list(second_gen)

        first_x = []
        second_x = []
        first_y = []
        second_y = []

        for x, y in first_batches:
            first_x = x.tolist()
            first_y = y.tolist()

        for x, y in second_batches:
            second_x = x.tolist()
            second_y = y.tolist()

        self.assertListEqual(first_x, second_x)
        self.assertListEqual(first_y, second_y)

    def test_generator_seed(self):

        # self.assertListEqual([np.array([1, 2, 3])], [np.array([1, 2, 3])])

        ### Seed check

        # Also, batch size 1 less than total
        self.try_test_seed(dataset=self.dataset_basic_tuple_10, batch_size=9, seed=101)

        # Batch size of 1
        self.try_test_seed(self.dataset_basic_tuple_10, batch_size=1, seed=0)

if __name__ == '__main__':
    unittest.main()
