# coding: utf-8

import sys

sys.path.append('..')
from common.trainer import Trainer
from common.optimizer import Adam
from ch03.simple_cbow import SimpleCBOW
from ch03.simple_skip_gram import SimpleSkipGram
from common.util import preprocess, create_contexts_target, convert_to_onehot

# 上下文查看 1 个单词
window_size = 1
# 转换为 size 为 5 的隐含层
hidden_size = 5
# 每轮处理的 context 和 target 数量
batch_size = 3
# 最大迭代轮数
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus)
target = convert_to_onehot(target, vocab_size)
contexts = convert_to_onehot(contexts, vocab_size)

model = SimpleSkipGram(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vec
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])
