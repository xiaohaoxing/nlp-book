# coding: utf-8
import sys

sys.path.append('..')
from common.util import preprocess, create_co_matrix, ppmi
import numpy as np
import matplotlib.pyplot as plt

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

# SGD 奇异值分解
U, S, V = np.linalg.svd(W)

print(C[0])
print(W[0])
print(U[0])
# 取出前 2 个元素作为维度
print(U[0, :2])

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()
