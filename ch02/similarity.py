# coding: utf-8
import sys

sys.path.append('..')
from common.util import preprocess, create_co_matrix, cos_similarity

import numpy as np

text = 'You say goodbye and I say hello.'

corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c_you = C[word_to_id['you']]
c_i = C[word_to_id['i']]

print(cos_similarity(c_you, c_i))
