import torch.nn as nn
import torch as torch

class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
    
    def forward(self, input):
        b, c, h , w = input.size()
        
        # Max Pool Height Dimension for variable height inputs
        f_pool = nn.MaxPool2d((h, 1), (1,1))
        conv = f_pool(input)
        b, c, h , w = conv.size()
        
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        
        return conv

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
 
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512] # size of each layer
        # nm = [64, 128, 256, 256, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut)) # reduces overfitting

            # always add a Relu as well (rectifies all negative values to 0, which trains it faster, helps to alleviate the vanishing gradient problem which is the issue where the lower layers of the network train very slowly because the gradient decreases exponentially through the layers")
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64 pooling is downsampling, with a "stride" overlap
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        
        self.cnn = cnn
        
        self.maxp = nn.Sequential(MaxPooling())
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features (cnn)
        conv = self.cnn(input)
        
        pooled = self.maxp(conv)

        # rnn features
        output = self.rnn(pooled)

        return output


use_cuda = torch.cuda.is_available()
from torch.autograd import Variable
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self,nc,nh,leakyRelu=False):# imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(EncoderRNN, self).__init__()
        # self.hidden_size = hidden_size

        self.hidden_size = nh
        # self.output_size = nc
        self.max_length = 100
        self.attn = nn.Linear(self.hidden_size , self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]  # size of each layer

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))  # reduces overfitting

            # always add a Relu as well (rectifies all negative values to 0, which trains it faster, helps to alleviate the vanishing gradient problem which is the issue where the lower layers of the network train very slowly because the gradient decreases exponentially through the layers")
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0),
                       nn.MaxPool2d(2, 2))  # 64x16x64 pooling is downsampling, with a "stride" overlap
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn

    def forward(self, input):

        conv = self.cnn(input)
        
        # Removing max pooling for attention
        
    # b, c, h, w = conv.size()

        # Max Pool Height Dimension for variable height inputs
        #f_pool = nn.MaxPool2d((h, 1), (1, 1))
        #conv = f_pool(conv)
        #b, c, h, w = conv.size()

        #assert h == 1, "the height of conv must be 1"
        #conv = conv.squeeze(2)
        #conv = conv.permute(2, 0, 1)  # [w, b, c]

        #output = conv
        # return output
        return conv

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

MAX_LENGTH = 500 #Usually 100 is find for small images. For bigger images, this parameter should be calibrated
class AttnDecoderRNN(nn.Module):
    def __init__(self, nh, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = nh
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(512 + self.hidden_size, self.hidden_size) #units from enconder + attn_applied
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)  # I think this will break with the variable image size, maybe. just needs to be
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class MTLM(nn.Module):
    def __init__(self):
        super(MTLM, self).__init__()
        self.task_combine = nn.Linear(2, 1) #units from enconder + attn_applied

    def forward(self, loss1, loss2):

        input = torch.cat((loss1, loss2), 0)
        y_predict = self.task_combine(input) #.unsqueeze(0)
        return y_predict

