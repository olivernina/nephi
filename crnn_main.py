from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
import io
encoding = 'utf-8'

import models.crnn as crnn

import sys  
stdout = sys.stdout
reload(sys)  
sys.setdefaultencoding('utf-8')
sys.stdout = stdout
from model_error import cer, wer

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', required=True, help='path to dataset')
parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for, default 25')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', default='', help="path to start crnn file (to continue training between invocations)")
parser.add_argument('--dataset', type=str, default='READ', help='type of dataset to use such as READ or ICFHR default is READ')
parser.add_argument('--experiment', default=None, help='Where to store samples and models (model save directory)')
parser.add_argument('--displayInterval', type=int, default=5, help='Interval number of batches to display progress')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display to console when test')
parser.add_argument('--valEpoch', type=int, default=10, help='Epoch to display validation and training error rates')
parser.add_argument('--saveEpoch', type=int, default=5, help='Epochs at which to save snapshot of model to experiment directory, ex: netCRNN_{1}_{2}.pth')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is false, rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is false, use rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--test_icfhr', action='store_true', help='Whether to make predictions on the test set according to ICFHR format')
parser.add_argument('--test_file', default='test_file', help='Path to file to store test set results')

opt = parser.parse_args()
print("Running with options:", opt)

if opt.experiment is None:
    opt.experiment = 'expr'
if not os.path.isdir(opt.experiment):
  os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000)  # fix seed (new random seed)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainroot)
assert train_dataset

test_dataset = dataset.lmdbDataset(root=opt.valroot)
assert test_dataset

minn = min(len(test_dataset), len(train_dataset))
if opt.batchSize > minn:
  print("Adjusting batchSize down for small test size to ", minn) # without this it does some tail sample thing wrong I think...
  opt.batchSize = minn

if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=opt.batchSize, sampler=dataset.randomSequentialSampler(test_dataset, opt.batchSize),
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

dataset_alphabet  = ''
if opt.dataset == 'READ':
    dataset_alphabet = 'alphabets/alphabet_read.txt'
elif opt.dataset =='ICFHR':
    dataset_alphabet = 'alphabets/ICFHR_alphabet.txt'
else:
    print('dataset '+opt.dataset+' not supported')
    sys.exit(0)

if os.path.exists(dataset_alphabet):
    alphabet = ''
    with io.open(dataset_alphabet, 'r', encoding=encoding) as myfile:
        alphabet = myfile.read().split()
        alphabet.append(u' ')
        alphabet = ''.join(alphabet)

    if len(alphabet)>1:
        opt.alphabet = alphabet

print("This is the alphabet:")
print(opt.alphabet)
nclass = len(opt.alphabet) + 1
nc = 1 # always one

converter = utils.strLabelConverter(opt.alphabet)
criterion = CTCLoss()

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
#print("Got to the weight initialization and loading pretrained model")
crnn.apply(weights_init)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgW, opt.imgH)   #  
text = torch.IntTensor(opt.batchSize * 5)          # RA: I don't understand why the text has this size
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

if opt.crnn != '':
    print('loading pretrained model from %s' % opt.crnn)
    crnn.load_state_dict(torch.load(opt.crnn))

print("Your neural network:", crnn)
    
image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr) # default

def test(net, dataset, criterion):
    print('Start test set predictions')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()

    test_iter = iter(dataset)
    all_file_names = []
    all_preds = []
    image_count = 0

    for i in range(len(dataset)):
        data = test_iter.next()
        #i += 1
        cpu_images, __, file_names = data
        batch_size = cpu_images.size(0)
        image_count = image_count + batch_size
        utils.loadData(image, cpu_images)
 
        preds = crnn(image)
        #print(preds.size())
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        
        
        # RA: While I am not sure yet, it looks like a greedy decoder and not beam search is being used here
        # Case is ignored in the accuracy, which is not ideal for an actual working system
        
        _, preds = preds.max(2)     
        if torch.__version__ < '0.2':
          preds = preds.squeeze(2) # https://github.com/meijieru/crnn.pytorch/issues/31
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        
        all_preds.extend(sim_preds)
        all_file_names.extend([f.partition(".jpg")[0] for f in file_names])

    
    print("Total number of images in test set: %8d" % image_count)
    
    return (all_file_names, all_preds)

def val(net, dataset, criterion, max_iter=1000):
    print('Start validation set')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    
    # RA: Testing out resizing 
    #data_loader = torch.utils.data.DataLoader(
    #    dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    #val_iter = iter(data_loader)
    val_iter = iter(dataset)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()
    
    image_count = 0
    # Character and word error rate lists
    char_error = []
    w_error = []

    max_iter = min(max_iter, len(dataset))
    
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts, __ = data
        batch_size = cpu_images.size(0)
        image_count = image_count + batch_size
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        #print(preds.size())
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)
        
        
        # RA: While I am not sure yet, it looks like a greedy decoder and not beam search is being used here
        # Case is ignored in the accuracy, which is not ideal for an actual working system
        
        _, preds = preds.max(2)
        if torch.__version__ < '0.2':
          preds = preds.squeeze(2) # https://github.com/meijieru/crnn.pytorch/issues/31
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target:
                n_correct += 1
            
            # Case-insensitive character and word error rates
            char_error.append(cer(pred, target))
            w_error.append(wer(pred, target))

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
    
    print("Total number of images in validation set: %8d" % image_count)
    
    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuracy: %f' % (loss_avg.val(), accuracy))
    
    char_arr = np.array(char_error)
    w_arr = np.array(w_error)
    print("Character error rate mean: %4.4f; Character error rate sd: %4.4f" % (np.mean(char_arr), np.std(char_arr, ddof=1)))
    print("Word error rate mean: %4.4f; Word error rate sd: %4.4f" % (np.mean(w_arr), np.std(w_arr, ddof=1)))
    
    return (char_error, w_error)


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts, __ = data
    
    # I think here is a place we could add dynamic data augmentation with each batch. We could also put it in the batch generation code if it is called dynamically
    
    
    
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


print("Starting training...")

for epoch in range(opt.niter):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        
        # Start by running prediction on test set, doing nothing else
        if opt.test_icfhr:
            files, predictions = test(crnn, test_loader, criterion)
            with io.open(opt.test_file, "w", encoding=encoding) as test_results:
                for file, pred in zip(files, predictions):
                    test_results.write(' '.join([file, pred]) + "\n")  # this should combine ascii text and unicode correctly
            break
        
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer) # it trains/backpropagates once/batch, each batch is made up of "batchSize" images
        # once you're done with all batches that's the end of one "epoch"
        loss_avg.add(cost)
        i += 1
        
        # Display the loss
        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' % (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()
        
        # Evaluate performance on validation and training sets periodically
        if (epoch % opt.valEpoch == 0) and (i >= len(train_loader)):      # Runs at end of some epochs
            val(crnn, test_loader, criterion)
            val(crnn, train_loader, criterion)

        # do checkpointing
        if (epoch % opt.saveEpoch == 0) and (i >= len(train_loader)):      # Runs at end of some epochs
            print("Saving epoch",  '{0}/netCRNN_{1}_{2}.pth'.format(opt.experiment, epoch, i))
            torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.experiment, epoch, i))
    
    if opt.test_icfhr:
        break
