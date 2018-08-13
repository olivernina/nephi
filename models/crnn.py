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