#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import collections

SOS_token = 1
EOS_token = 2

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False,attention=False):

        self._ignore_case = ignore_case
        self.attention = attention

        if self._ignore_case:
            alphabet = alphabet.lower()

        if attention:
            self.offset = 3
            self.alphabet = alphabet + u'-@#'  # for `-3` index
        else:
            self.offset = 1
            self.alphabet = alphabet + u'-'  # for `-1` index

        self.num_classes = len(self.alphabet)

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by warp_ctc
            # Oliver: I am proposing to reserve 1 and 2 for SOS and EOS
            self.dict[char] = i + self.offset

        if attention:
            self.dict['#']=EOS_token

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, basestring):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]

            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            if self.attention:
                # length = [len(s) for s in text]
                length = [len(s)+1 for s in text]
                text = '#'.join(text)
                text= text+'#'
            else:
                length = [len(s) for s in text]
                text = ''.join(text)

            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - self.offset] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (t[i] != EOS_token if self.attention else True) and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - self.offset])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):

        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()


        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img


# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import datetime
import os
import numpy as np
#
# def showPlot(points,prefix):
#     plt.interactive(False)
#     plt.figure()
#     fig, ax = plt.subplots()
#     # this locator puts ticks at regular intervals
#     loc = ticker.MultipleLocator(base=0.2)
#     ax.yaxis.set_major_locator(loc)
#     plt.plot(points)
#
#     if not os.path.exists('./plots'):
#         os.mkdir('./plots')
#
#     plt.savefig('plots/' +prefix+'_'+ datetime.datetime.now().strftime("%m-%d-%y-%H-%M") + '.png')

def savePlot(history, res_dir):
    plot_path = os.path.join(res_dir, 'plot.txt')
    np.savetxt(plot_path,
                  history, fmt='%.3f')

def parse_model_name(opt):
    epoch = opt.pre_model.split('_')[-2]
    i = opt.pre_model.split('_')[-1].split('.')[0]
    pre_dir = opt.pre_model.split('net')[0]
    return epoch,i,pre_dir