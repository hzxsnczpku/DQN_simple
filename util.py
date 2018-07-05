import random
import numpy as np
from collections import deque, defaultdict
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter


USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
INT = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


class Memory(object):
    def __init__(self, capacity):
        self.mem = deque(maxlen=capacity)

    def append(self, transition):
        self.mem.append(transition)

    def sample(self, batch_size):
        samples = random.sample(self.mem, batch_size)
        return map(np.array, zip(*samples))


class ResultsBuffer(object):
    def __init__(self, base_path):
        self.buffer = defaultdict(list)
        self.index = defaultdict(int)
        self.summary_writer = SummaryWriter(base_path)

    def update_info(self, info):
        for key in info:
            self.summary_writer.add_scalar(key, info[key], self.index[key])
            self.index[key] += 1


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    var = Variable(torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad).type(dtype)
    return var.cuda() if USE_CUDA else var
