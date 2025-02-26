import numpy as np

from Functions import Softmax

# MNIST example (let final layer be 10 cols, let 2nd final layer be 16):
        #   If softmax is activation function for layer,
        #   dO_dz is n x 10 x 10, rather than usual n x 10
        #   In special case, want to treat delta as n x (1 x 10), rather than n x 10,
        #   then collapse back to n x 10, so rest of backprop works.
        #   Can use einsum to represent this logic more compactly.


sm_in = np.arange(30).reshape(-1, 10)
softmax_d = Softmax().derivative(sm_in)

manual_derivative = np.zeros((3, 10, 10))
for b in range(3):
    manual_derivative[b] = Softmax().single_sample_derivative(sm_in[b])

ers = np.allclose(softmax_d, manual_derivative)

delta = np.arange(30).reshape(-1, 10)

usually_a_hadamard_term = np.einsum("bj, bjk -> bk", delta, softmax_d)

temp = np.zeros((3, 10))
for b in range(3):
    brow = delta[b]
    softmax_d_mat = softmax_d[b]
    temp[b] = np.matmul(brow, softmax_d_mat)

what = np.allclose(temp, usually_a_hadamard_term)

print(usually_a_hadamard_term)
