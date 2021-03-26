# coding: utf-8

import numpy as np

# 微小量
eps = 1e-8


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


####################
#  语料库处理相关函数  #
####################

def preprocess(text: str):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


# 生成共现矩阵
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


# 余弦相似度
def cos_similarity(x, y):
    nx = x / np.sqrt(np.sum(x ** 2) + eps)
    ny = y / np.sqrt(np.sum(y ** 2) + eps)
    return np.dot(nx, ny)


# 获取一个单词最相似的词
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print('%s 不在词典中' % query)
        return

    print('\n[query]' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def ppmi(C, verbose=False, eps=1e-8):
    '''
    计算整个矩阵的 PPMI（正的点互信息），得到一个PPMI 矩阵

    :param C: 共现矩阵
    :param verbose: 是否打印运行进度信息
    :param eps: 微小量
    :return: ppmi 矩阵
    '''
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100 + 1) == 0:
                    print('%.1f%% done' % (100 * cnt / total))
    return M


# 从语料库生成word2vec需要的上下文和目标词
def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size: -window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


# 将一维或者二维的数组转换为 onehot 形式（会扩充一维 size）
def convert_to_onehot(corpus, vocab_size):
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
        return one_hot
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1
        return one_hot
    else:
        return None


def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)


def query_vec_similarity(query_vec, word_to_id, id_to_word, word_vecs, top=5):
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    test = cos_similarity(word_vecs[word_to_id['better']], query_vec)
    print('target with better:' + str(test))
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_vecs[i], query_vec)
    count = 0
    for i in (-1 * similarity).argsort():
        print(' %s: %s' % (id_to_word[i], similarity[i]))
        count += 1
        if count >= top:
            return


def analogy(src1, tar1, src2, word_to_id, id_to_word, word_vecs, top=5):
    src1_id = word_to_id[src1]
    tar1_id = word_to_id[tar1]
    src2_id = word_to_id[src2]
    query_vec = word_vecs[src2_id] + word_vecs[tar1_id] - word_vecs[src1_id]
    print('\n[query]%s to %s like %s to what?' % (src1, tar1, src2))
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_vecs[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if i != src1_id and i != tar1_id and i != src2_id:
            print(' %s: %s' % (id_to_word[i], similarity[i]))

            count += 1
            if count >= top:
                return


def analogy_middle(left, right, word_to_id, id_to_word, word_vecs, top=5):
    left_id = word_to_id[left]
    right_id = word_to_id[right]
    query_vec = (word_vecs[left_id] + word_vecs[right_id]) / 2
    print('find word between %s and %s' % (left, right))
    query_vec_similarity(query_vec, word_to_id, id_to_word, word_vecs, top)
