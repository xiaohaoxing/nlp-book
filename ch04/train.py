# coding: utf-8

import sys

sys.path.append('..')
import numpy as np
from common.layers import Embedding, SigmoidWithLoss
import pickle
from common.trainer import Trainer
from dataset import ptb
from ch04.cbow import CBOW
from common.optimizer import Adam
from common.util import create_contexts_target, to_cpu, to_gpu
from common import config

##############################
# 使用 GPU 运行时需要设置为 True #
# config.GPU = True
##############################

# 设定超参数
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

# 读取数据
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

context, target = create_contexts_target(corpus, window_size)

if config.GPU:
    context, target = to_gpu(context), to_gpu(target)

# 构建模型
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 开始训练
trainer.fit(context, target, max_epoch, batch_size)
trainer.plot()

# 保存模型
word_vecs = model.word_vecs
if config.GPU:
    word_vecs = to_cpu(word_vecs)

params = {}
params['word_vecs'] = word_vecs
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)
