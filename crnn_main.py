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
import PIL
import os
import utils
import dataset
import io
from collections import Counter
encoding = 'utf-8'

import models.crnn
import torch.nn as nn

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
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for, default 25')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', default='', help="path to start crnn file (to continue training between invocations)")
parser.add_argument('--dataset', type=str, default='READ', help='type of dataset to use such as READ or ICFHR default is READ')
parser.add_argument('--displayInterval', type=int, default=100, help='Interval number of batches to display progress')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display to console when test')
parser.add_argument('--valEpoch', type=int, default=5, help='Epoch to display validation and training error rates')
parser.add_argument('--saveEpoch', type=int, default=5, help='Epochs at which to save snapshot of model to experiment directory, ex: netCRNN_{1}_{2}.pth')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is false, rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is false, use rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--test_icfhr', action='store_true', help='Whether to make predictions on the test set according to ICFHR format')
parser.add_argument('--test_file', default='results/results.txt', help='Path to file to store test set results')
parser.add_argument('--test_aug', action="store_true", help='Whether to use data augmentation at validation/test time')
parser.add_argument('--n_aug', type=int, default=20, help='Number of times to augment each image at validation/test time')
parser.add_argument('--binarize', action="store_true", help='Whether to use howe and sauvola binarization as separate channels, requires these data to already be in the lmdb databases')
parser.add_argument('--plot', action='store_true', help='Save plots')
parser.add_argument('--model', type=str, default='ctc', help='type of model used i.e. ctc, attention, attention+ctc')
parser.add_argument('--debug', action='store_true', help='Runs debug mode with 1000 samples of training')
parser.add_argument('--rdir', default='results', help='Where to store samples, models and plots (model save directory)')
parser.add_argument('--transform', action="store_true", help='Allow transformation of images')
parser.add_argument('--mode', type=str, default='train', help='i.e train, test. Mode of executing code')
parser.add_argument('--data_aug', action="store_true", help='Whether to use data augmentation')
parser.add_argument('--pre_model', default='', help="path to the pretrained model. For other models besides ctc just include one of the pretrained models")
parser.add_argument('--grid_distort', action="store_true", help='Whether to use grid distortion data augmentation')
parser.add_argument('--aug_thresh', type=float, default=1.0, help='Percent of samples to augment if any data augmentation is selected')
parser.add_argument('--rescale', action="store_true", help='Whether to use rescaling data augmentation')
parser.add_argument('--rescale_dim_up', type=float, default=1.0, help='increasing rescaling dimension for data augmentation')
parser.add_argument('--rescale_dim_down', type=float, default=1.0, help='decreasing rescaling dimension for data augmentation')
parser.add_argument('--mtlm', action='store_true', help='learning loss weights')

opt = parser.parse_args()
print("Running with options:", opt)

if not os.path.isdir(opt.rdir):
    os.system('mkdir {0}'.format(opt.rdir))

model_rpath = os.path.join(opt.rdir, opt.model)

if not os.path.exists(model_rpath):
    os.system('mkdir {0}'.format(model_rpath))
else:
    print('result directory {0} already exists'.format(model_rpath))
    # if not opt.debug:
    #     sys.exit(0)
    # os.system('rm {0}/*'.format(model_res_path))

opt.manualSeed = random.randint(1, 10000)  # fix seed (new random seed)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


print("Used rotation and shearing of training images?")
print(opt.transform)

deg = 10
shear = (-5, 5)
if opt.transform:
    print("Used degree for rotation of images")
    print(deg)
    print("Used shear on images")
    print(shear)

augment = opt.grid_distort
rescale= opt.rescale
print("Used Grid Distortion augmentation?")
print(augment)
print("Rescale images randomly?")
print(rescale)
scale = (opt.rescale_dim_down, opt.rescale_dim_up)
if rescale:
    print("Scale multiplication used: (down x, up x)")
    print(scale)


if opt.transform:
    from torchvision.transforms import RandomAffine
    lin_transform = RandomAffine(deg, shear=shear, resample=PIL.Image.BILINEAR, fillcolor="white")
else:
    lin_transform = None

train_dataset = dataset.lmdbDataset(root=opt.trainroot, binarize = opt.binarize, augment=augment, scale=rescale, dataset=opt.dataset, test=opt.test_icfhr, transform= lin_transform, debug=opt.debug, scale_dim = scale, thresh = opt.aug_thresh)

assert train_dataset

test_dataset = dataset.lmdbDataset(root=opt.valroot, binarize=opt.binarize, test=opt.test_icfhr, augment=augment if opt.test_aug else False,
                                  transform = lin_transform if opt.test_aug else None, scale = rescale if opt.test_aug else False, scale_dim = scale if opt.test_aug else 1.0, thresh = opt.aug_thresh)
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
    train_dataset, batch_size=opt.batchSize, shuffle=True, #sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=opt.batchSize, shuffle=True,  #sampler=dataset.randomSequentialSampler(test_dataset, opt.batchSize),
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

dataset_alphabet  = ''
opt.alphabet = ''
if opt.dataset == 'READ':
    dataset_alphabet = 'alphabets/READ_alphabet.txt'
elif opt.dataset =='ICFHR':
    dataset_alphabet = 'alphabets/ICFHR_alphabet.txt'
elif opt.dataset == 'JOURNAL_ICFHR':
    dataset_alphabet = 'alphabets/JOURNAL_ICFHR_alphabet.txt'
elif opt.dataset == "JOURNAL_ICFHR_IAM":
    dataset_alphabet = 'alphabets/JOURNAL_ICFHR_IAM_alphabet.txt'
elif opt.dataset == "JOURNAL_WHOLE":
    dataset_alphabet = 'alphabets/JOURNAL_WHOLE_alphabet.txt'
else:
    print('dataset '+opt.dataset+' not supported')
    sys.exit(0)

if os.path.exists(dataset_alphabet):
    alphabet = ''
    with io.open(dataset_alphabet, 'r', encoding=encoding) as myfile:
        alphabet = myfile.read().split()
        alphabet.append(u' ')
        # alphabet = set(alphabet) # This was a lazy line for not providing unique characters in an alphabet for the russell private dataset. if present, it makes all our previous models not work.
        alphabet = ''.join(alphabet)

    if len(alphabet)>1:
        opt.alphabet = alphabet

print("This is the alphabet:")
print(opt.alphabet)

opt.model = 'ctc'

if opt.model == 'ctc':
    converter = utils.strLabelConverter(opt.alphabet)


nclass = converter.num_classes
nc = 3 if opt.binarize else 1

if opt.model=='ctc':
    criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if opt.model=='ctc':
    crnn = models.crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
    #print("Got to the weight initialization and loading pretrained model")
    crnn.apply(weights_init)


image = torch.FloatTensor(opt.batchSize, 3 if opt.binarize else 1, opt.imgW, opt.imgH)   #
text = torch.IntTensor(opt.batchSize * 5)          # RA: I don't understand why the text has this size
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    if opt.model=='ctc':
        crnn.cuda()
        crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
        criterion = criterion.cuda()

    image = image.cuda()


if opt.pre_model != '':
    if opt.model=='ctc':
        print('loading pretrained model from %s' % opt.pre_model)
        pre_model = torch.load(opt.pre_model)
        crnn.load_state_dict(pre_model)

elif opt.mode == "test":
    print("Pretrained model directory should be provided for testing mode.")
    os.exit(0)


if opt.model=='ctc':
    print("Your neural network:", crnn)


image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()



if opt.model=='ctc':
    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)  # default


def test(net, dataset, criterion, n_aug=1):
    print('Start test set predictions')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()

    
    all_file_names = []
    all_preds = []
    image_count = 0
    pred_dict = {}
    
    for epoch in range(n_aug):
        test_iter = iter(dataset)
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

            for pred, f in zip(sim_preds, file_names):
                if f not in pred_dict:
                    pred_dict[f] = [pred]
                else:
                    pred_dict[f].append(pred)

    for f, final_preds in pred_dict.items():
        all_preds.append(Counter(final_preds).most_common(1)[0][0])
        all_file_names.append(f.partition(".jpg")[0])

    
    print("Total number of images in test set: %8d" % image_count)
    
    return (all_file_names, all_preds)



def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts, __ = data

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



def val(net, dataset, criterion, max_iter=1000, test_aug=False, n_aug=1):

    print('Start validation set')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    image_count = 0
    # Character and word error rate lists
    char_error = []
    w_error = []
    
    pred_dict = {}
    gt_dict = {}

    for epoch in range(n_aug):
        max_iter = len(dataset) if test_aug else min(max_iter, len(dataset))
        val_iter = iter(dataset)
   
        for i in range(max_iter):
            data = val_iter.next()
            i += 1
            cpu_images, cpu_texts, cpu_files = data
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

            for pred, target, f in zip(sim_preds, cpu_texts, cpu_files):
                if f not in gt_dict:
                    gt_dict[f] = target
                    pred_dict[f] = []
                pred_dict[f].append(pred)
                if pred == target:
                    n_correct += 1
            
    # Case-sensitive character and word error rates
    for f, target in gt_dict.items():
        # Finds the most commonly predicted string for all the augmented images
        best_pred = Counter(pred_dict[f]).most_common(1)[0][0]
        char_error.append(cer(best_pred, target))
        w_error.append(wer(best_pred, target))

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    print("Total number of images in validation set: %8d" % image_count)

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuracy: %f' % (loss_avg.val(), accuracy))

    char_arr = np.array(char_error)
    w_arr = np.array(w_error)
    char_mean_error = np.mean(char_arr)
    word_mean_error = np.mean(w_arr)

    print("Character error rate mean: %4.4f; Character error rate sd: %4.4f" % (
    char_mean_error, np.std(char_arr, ddof=1)))
    print("Word error rate mean: %4.4f; Word error rate sd: %4.4f" % (word_mean_error, np.std(w_arr, ddof=1)))

    return char_mean_error, word_mean_error, accuracy


def setupTrain(net):
    for p in net.parameters():
        p.requires_grad = True
        net.train()



if opt.mode =='train':

    print("Starting training...")

    history_errors = []
    loss = 0


    for epoch in range(opt.niter):
        train_iter = iter(train_loader)
        i = 1

        while i < len(train_loader):
            if opt.model=='ctc':
                for p in crnn.parameters():
                    p.requires_grad = True
                crnn.train()

                loss = trainBatch(crnn, criterion, optimizer) # it trains/backpropagates once/batch, each batch is made up of "batchSize" images

            # once you're done with all batches that's the end of one "epoch"
            loss_avg.add(loss)
            i += 1

            # Display the loss
            if i % opt.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' % (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
                if loss_avg.val() <100000000:
                    loss = loss_avg.val()

                loss_avg.reset()

            # Evaluate performance on validation and training sets periodically
            if (epoch % opt.valEpoch == 0) and (i >= len(train_loader)):      # Runs at end of some epochs

                if opt.model=='ctc':
                    val_CER, val_WER, val_ACC = val(crnn, test_loader, criterion, test_aug = opt.test_aug, n_aug = opt.n_aug if opt.test_aug else 1)
                    train_CER, train_WER, train_ACC = val(crnn, train_loader, criterion)

                history_errors.append([epoch, i, loss,train_ACC,train_WER,train_CER,val_ACC,val_WER,val_CER])

                if opt.plot:
                    utils.savePlot(history_errors,model_rpath)

            # do checkpointing
            if (epoch % opt.saveEpoch == 0) and (i >= len(train_loader)):      # Runs at end of some epochs
                print("Saving epoch",  '{0}/netCRNN_{1}_{2}.pth'.format(model_rpath, epoch, i))
     
                if opt.model=='ctc':
                    torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(model_rpath, epoch, i))


elif opt.mode=='test':
    if opt.dataset=='ICFHR':
        if opt.model=='ctc':
            files, predictions = test(crnn, test_loader, criterion, n_aug=opt.n_aug if opt.test_aug else 1)
        with io.open(opt.test_file, "w", encoding=encoding) as test_results:
            for f, pred in zip(files, predictions):
                test_results.write(' '.join([unicode(f, encoding=encoding),
                                             pred]) + u"\n")  # this should combine ascii text and unicode correctly
    elif opt.dataset=='READ':
        if opt.model=='ctc':
            files, predictions = test(crnn, test_loader, criterion, n_aug=opt.n_aug if opt.test_aug else 1)
            with io.open(opt.test_file, "w", encoding=encoding) as test_results:
                for f, pred in zip(files, predictions):
                    test_results.write(' '.join([unicode(f, encoding=encoding),
                                                 pred]) + u"\n")  # this should combine ascii text and unicode correctly