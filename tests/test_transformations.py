import sys
sys.path.append('/home')

import numpy as np
from pavlov.pipeline import transformations

def test_add_dim():
    state = np.ones((3,3))
    results = [transformations.add_dim(i)(state, 0) for i in range(-1, 3)]
    expected = [
        np.array([[[1],[1],[1]], [[1],[1],[1]], [[1],[1],[1]]]),
        np.array([[[1,1,1], [1,1,1], [1,1,1]]]),
        np.array([[[1,1,1]], [[1,1,1]], [[1,1,1]]]),
        np.array([[[1],[1],[1]], [[1],[1],[1]], [[1],[1],[1]]])
    ]
    assert np.array_equal(results[0], expected[0])
    assert np.array_equal(results[1], expected[1])
    assert np.array_equal(results[2], expected[2])
    assert np.array_equal(results[3], expected[3])


def test_one_hot():
    oneD_state = np.array([1,4,3,6,2])
    twoD_state = np.array([[1,4,3], [6,2,5]])
    oneD_result = transformations.one_hot(6, 1)(oneD_state, 0)
    twoD_result = transformations.one_hot(6, 1)(twoD_state, 0)
    oneD_expected = np.array([[1,0,0,0,0,0],[0,0,0,1,0,0], [0,0,1,0,0,0],
                             [0,0,0,0,0,1], [0,1,0,0,0,0]])
    twoD_expected = np.array([[[1,0,0,0,0,0], [0,0,0,1,0,0], [0,0,1,0,0,0]],
                              [[0,0,0,0,0,1], [0,1,0,0,0,0], [0,0,0,0,1,0]]])
    assert np.array_equal(oneD_result, oneD_expected)
    assert np.array_equal(twoD_result, twoD_expected)


def test_rgb_to_binary():
    state = np.array([[[0, 1, 0], [0, 0, 2], [0, 0, 0]],
                      [[0, 0, 3], [0, 0, 0], [0, 0, 0]],
                      [[4, 0, 0], [5, 0, 0], [0, 0, 0]]])
    result = transformations.rgb_to_binary()(state, 0)
    expected = np.array([[[1],[1],[0]], [[1], [0], [0]], [[1], [1], [0]]])
    assert np.array_equal(result, expected)


class TestStackConsecutive:
    def setup(self):
        self.stacker = transformations.stack_consecutive(2)

    def test_stack_consecutive(self):
        state1 = np.ones((3,4)) * 2
        state2 = np.ones((3,4)) * 3
        result1 = self.stacker(state1, 0)
        result2 = self.stacker(state2, 0)
        expected1 = np.array([[[2,2,2,2], [2,2,2,2], [2,2,2,2]],
                              [[0,0,0,0], [0,0,0,0], [0,0,0,0]]])
        expected2 = np.array([[[3,3,3,3], [3,3,3,3], [3,3,3,3]],
                              [[2,2,2,2], [2,2,2,2], [2,2,2,2]]])
        assert np.array_equal(result1, expected1)
        assert np.array_equal(result2, expected2)


class TestCombineConsecutive:
    def setup(self):
        self.combiner = transformations.combine_consecutive(2, 'max')

    def test_combine_consecutive(self):
        state1 = np.ones((2,2)) * 2
        state2 = np.ones((2,2)) * 3
        result1 = self.combiner(state1, 0)
        result2 = self.combiner(state2, 0)
        expected1 = np.array([[2,2], [2,2]])
        expected2 = np.array([[3,3], [3,3]])
        assert np.array_equal(result1, expected1)
        assert np.array_equal(result2, expected2)
