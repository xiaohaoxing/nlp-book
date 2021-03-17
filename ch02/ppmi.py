# coding: utf-8
import sys

sys.path.append('..')
from common.util import preprocess, create_co_matrix, cos_similarity, ppmi
import numpy as np

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

# 精确到 3 位小数
np.set_printoptions(precision=3)

print('covariance matrix')
print(C)
print('-' * 50)
print('PPMI')
print(W)
