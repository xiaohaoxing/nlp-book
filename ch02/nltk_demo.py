import nltk
# if nltk.corpus is None or nltk.corpus.wordnet is None:
# nltk.download()
from nltk.corpus import wordnet

# 打印所有同义词词簇
print(wordnet.synsets('car'))

# 打印指定的一个词义的定义
car = wordnet.synset('car.n.01')
print(car.definition())

# 打印一个词的所有同义词
print(car.lemma_names())

# 获取 car 的一条路径(可能存在多条路径到顶）
path = car.hypernym_paths()[0]
print(path)

novel = wordnet.synset('novel.n.01')
dog = wordnet.synset('dog.n.01')
motorcycle = wordnet.synset('motorcycle.n.01')

simi1 = car.path_similarity(novel)
simi2 = car.path_similarity(dog)
simi3 = car.path_similarity(motorcycle)
print('car\'s similarity with novel is:%.2f' % simi1)
print('car\'s similarity with dog is:%.2f' % simi2)
print('car\'s similarity with motorcycle is:%.2f' % simi3)
