# coding: utf-8

import sys

sys.path.append('..')
from common.util import most_similar, analogy, analogy_middle
import pickle

pkl_file = 'pre_cbow_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']
    print(word_vecs[word_to_id['better']])
    # queries = ['you', 'year', 'car', 'toyota']
    # for query in queries:
    #     most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
    # 性别推论
    analogy('man', 'king', 'woman', word_to_id, id_to_word, word_vecs, top=5)
    # 过去式推论
    analogy('take', 'took', 'go', word_to_id, id_to_word, word_vecs, top=5)
    # 复数推论
    analogy('car', 'cars', 'child', word_to_id, id_to_word, word_vecs, top=5)
    # 比较级推论
    analogy('good', 'better', 'bad', word_to_id, id_to_word, word_vecs, top=5)
    # 两个词中间词
    analogy_middle('good', 'best', word_to_id, id_to_word, word_vecs, top=5)
    # better 并不在 good 和 best 中间位置
    print('good:', word_vecs[word_to_id['good']])
    print('better:', word_vecs[word_to_id['better']])
    print('best:', word_vecs[word_to_id['best']])
