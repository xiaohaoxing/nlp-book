import numpy as np
from common.util import clip_grads

dW1 = np.random.rand(3, 3) * 10
dW2 = np.random.rand(3, 3) * 10
grads = [dW1, dW2]
max_norm = 5.0

clip_grads(grads, max_norm)
