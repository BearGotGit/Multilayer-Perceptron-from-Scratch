import numpy as np


dropout_generator = np.random.Generator(np.random.PCG64(69))

O = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
dropout_probability = 0.5


O = np.multiply(O, dropout_generator.binomial(1, 1 - dropout_probability, O.shape))

O = np.multiply(O, dropout_generator.binomial(1, 1 - dropout_probability, O.shape))


inp = np.ones_like((2,3)) * 1e-10
are_you_sure = np.log(inp)

inp = np.ones_like((2,3)) * 1e2
are_you_sure = np.exp(inp)


print(O)