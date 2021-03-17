import sys

sys.path.append('..')

from dataset import spiral

x, t = spiral.load_data()
print('x', x.shape)
print('t', t.shape)
