# coding:utf-8

import sys

sys.path.append('..')
from common.util import create_contexts_target, preprocess, convert_to_onehot

text = 'You say goodbye and I say hello.'

corpus, word_to_id, id_to_word = preprocess(text)
contexts, target = create_contexts_target(corpus)

print(contexts)
print(target)

vocab_size = len(word_to_id)
contexts = convert_to_onehot(contexts, vocab_size)
target = convert_to_onehot(target, vocab_size)
print(contexts)
print(target)
