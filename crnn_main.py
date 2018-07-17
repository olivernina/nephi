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
parser.add_argument('--lr_att', type=float, default=0.1, help='learning rate for Attention model, default=0.00005')
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
parser.add_argument('--rescale', action="store_true", help='Whether to use rescaling data augmentation')
parser.add_argument('--rescale_dim', type=float, default=1.0, help='rescaling dimension for data augmentation')
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


    # RA: The next augmentation should be just 5 degree rotation, 5 degree shear, the 60 is probably overkill; other publications use 5 for both
deg = 5
shear = (-20, 20)
print("Used degree for rotation of images")
print(deg)
print("Used shear on images")
print(shear)

augment = opt.grid_distort
rescale= opt.rescale
print("Use Grid Distortion augmentation?")
print(augment)
print("Rescale images randomly?")
print(rescale)
scale = opt.rescale_dim
print("Scale multiplication used:")
print(scale)

if opt.transform:
    from torchvision.transforms import RandomAffine
    lin_transform = RandomAffine(deg, shear=shear, resample=PIL.Image.BILINEAR, fillcolor="white")
else:
    lin_transform = None

train_dataset = dataset.lmdbDataset(root=opt.trainroot, binarize = opt.binarize, augment=augment, scale=rescale, dataset=opt.dataset, test=opt.test_icfhr, transform= lin_transform, debug=opt.debug, scale_dim = scale)

assert train_dataset

test_dataset = dataset.lmdbDataset(root=opt.valroot, binarize=opt.binarize, test=opt.test_icfhr, augment=augment if opt.test_aug else False,
                                  transform = lin_transform if opt.test_aug else None, scale = rescale if opt.test_aug else False, scale_dim = scale if opt.test_aug else 1.0)
assert test_dataset

minn = min(len(test_dataset), len(train_dataset))
if opt.batchSize > minn:
  print("Adjusting batchSize down for small test size to ", minn) # without this it does some tail sample thing wrong I think...
  opt.batchSize = minn

if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None

if opt.model == 'attention' :
    opt.batchSize = 1

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


if opt.model == 'ctc':
    converter = utils.strLabelConverter(opt.alphabet,attention=False)
else:
    converter = utils.strLabelConverter(opt.alphabet, attention=True)

nclass = converter.num_classes
nc = 3 if opt.binarize else 1

if opt.model=='ctc':
    criterion = CTCLoss()
elif opt.model=='attention':
    criterion = torch.nn.NLLLoss()
elif opt.model=='attention+ctc':
    criterion_ctc = CTCLoss()
    criterion_att = torch.nn.NLLLoss()
    if opt.mtlm:
        criterion_mtlm =torch.nn.MSELoss(size_average=False)
elif opt.model=='ctc_pretrain':
    criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if opt.model=='attention':
    encoder = models.crnn.EncoderRNN(nc,opt.nh,nclass)
    attn_decoder = models.crnn.AttnDecoderRNN(opt.nh, nclass, dropout_p=0.1)
    encoder.apply(weights_init)
elif opt.model=='ctc':
    crnn = models.crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
    #print("Got to the weight initialization and loading pretrained model")
    crnn.apply(weights_init)
elif opt.model=='attention+ctc':
    encoder_ctc = models.crnn.EncoderRNN(nc,opt.nh)
    encoder_ctc.apply(weights_init)
    decoder_att = models.crnn.AttnDecoderRNN(opt.nh, nclass, dropout_p=0.1)
    decoder_ctc = nn.Sequential(models.crnn.BidirectionalLSTM(512, opt.nh, opt.nh),models.crnn.BidirectionalLSTM(opt.nh, opt.nh, nclass))

    if opt.mtlm:
        mtlm = models.crnn.MTLM()
elif opt.model=='ctc_pretrain':
    encoder_ctc = models.crnn.EncoderRNN(nc, opt.nh)
    encoder_ctc.apply(weights_init)
    decoder_att = models.crnn.AttnDecoderRNN(opt.nh, nclass, dropout_p=0.1)
    decoder_ctc = nn.Sequential(models.crnn.BidirectionalLSTM(512, opt.nh, opt.nh),
                                models.crnn.BidirectionalLSTM(opt.nh, opt.nh, nclass))


image = torch.FloatTensor(opt.batchSize, 3 if opt.binarize else 1, opt.imgW, opt.imgH)   #
text = torch.IntTensor(opt.batchSize * 5)          # RA: I don't understand why the text has this size
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    if opt.model=='attention':
        encoder.cuda()
        attn_decoder.cuda()
        criterion = criterion.cuda()
    elif opt.model=='ctc':
        crnn.cuda()
        crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
        criterion = criterion.cuda()
    elif opt.model=='attention+ctc':
        encoder_ctc.cuda()
        decoder_att.cuda()
        decoder_ctc.cuda()
        criterion_ctc = criterion_ctc.cuda()
        criterion_att = criterion_att.cuda()

        if opt.mtlm:
            mtlm.cuda()
            criterion_mtlm = criterion_mtlm.cuda()
    elif opt.model=='ctc_pretrain':
        encoder_ctc.cuda()
        encoder_ctc = torch.nn.DataParallel(encoder_ctc, device_ids=range(opt.ngpu))
        decoder_ctc.cuda()
        decoder_ctc = torch.nn.DataParallel(decoder_ctc, device_ids=range(opt.ngpu))
        criterion = criterion.cuda()

    image = image.cuda()


if opt.pre_model != '':
    if opt.model=='ctc':
        print('loading pretrained model from %s' % opt.pre_model)
        pre_model = torch.load(opt.pre_model)
        crnn.load_state_dict(pre_model)

    elif opt.model=='ctc_pretrain':
        epoch, i, pre_dir = utils.parse_model_name(opt)
        encoder_path = os.path.join(pre_dir,'netCNN_{0}_{1}.pth'.format(epoch,i))
        decoder_path = os.path.join(pre_dir, 'netCTCDec_{0}_{1}.pth'.format(epoch, i))

        print('loading pretrained model from %s' % encoder_path)
        pre_encoder = torch.load(encoder_path)
        encoder_ctc.load_state_dict(pre_encoder)

        print('loading pretrained model from %s' % decoder_path)
        pre_decoder = torch.load(decoder_path)
        decoder_ctc.load_state_dict(pre_decoder)

    elif opt.model=='attention+ctc':
        epoch = opt.pre_model.split('_')[-2]
        i = opt.pre_model.split('_')[-1].split('.')[0]
        pre_dir = opt.pre_model.split('net')[0]
        encoder_path = os.path.join(pre_dir,'netCNN_{0}_{1}.pth'.format(epoch,i))
        decoder_ctc_path = os.path.join(pre_dir, 'netCTCDec_{0}_{1}.pth'.format(epoch, i))
        decoder_att_path = os.path.join(pre_dir, 'netAttnDec_{0}_{1}.pth'.format(epoch, i))

        print('loading pretrained model from %s' % encoder_path)
        pre_encoder = torch.load(encoder_path)
        encoder_ctc.load_state_dict(pre_encoder)

        print('loading pretrained model from %s' % decoder_ctc_path)
        pre_decoder_ctc = torch.load(decoder_ctc_path)
        decoder_ctc.load_state_dict(pre_decoder_ctc)

        print('loading pretrained model from %s' % decoder_att_path)
        pre_decoder_att = torch.load(decoder_att_path)
        decoder_att.load_state_dict(pre_decoder_att)

    elif opt.model=='attention':
        epoch, i, pre_dir = utils.parse_model_name(opt)
        encoder_path = os.path.join(pre_dir, 'netCNN_{0}_{1}.pth'.format(epoch, i))
        decoder_att_path = os.path.join(pre_dir, 'netAttnDec_{0}_{1}.pth'.format(epoch, i))

        print('loading pretrained model from %s' % encoder_path)
        pre_encoder = torch.load(encoder_path)
        encoder.load_state_dict(pre_encoder)

        print('loading pretrained model from %s' % decoder_att_path)
        pre_decoder_att = torch.load(decoder_att_path)
        attn_decoder.load_state_dict(pre_decoder_att)

elif opt.mode == "test":
    print("Pretrained model directory should be provided for testing mode.")
    os.exit(0)


if opt.model=='attention':
    print("Your encoder network:", encoder)
    print("Your decoder network:", attn_decoder)
elif opt.model=='ctc':
    print("Your neural network:", crnn)
elif opt.model=='attention+ctc':
    print("Your encoder network:", encoder_ctc)
    print("Your att decoder network:", decoder_att)
    print("Your ctc decoder network:", decoder_ctc)
    if opt.mtlm:
        print("Your mtlm network:", mtlm)

elif opt.model=='ctc_pretrain':
    print("Your neural network:", encoder_ctc)
    print("Your neural network:", decoder_ctc)
    print("Your att decoder network:", decoder_att)

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

if opt.model=='attention':
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=opt.lr)
    decoder_optimizer = optim.SGD(attn_decoder.parameters(), lr=opt.lr)
elif opt.model=='ctc':
    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)  # default
elif opt.model=='attention+ctc':
    enc_ctc_optimizer = optim.RMSprop(encoder_ctc.parameters(), lr=opt.lr)
    dec_att_optimizer = optim.SGD(decoder_att.parameters(), lr=opt.lr_att)
    dec_ctc_optimizer = optim.RMSprop(decoder_ctc.parameters(), lr=opt.lr)

    if opt.mtlm:
        mtlm_optimizer = optim.RMSprop(mtlm.parameters(), lr=0.0001)

elif opt.model=='ctc_pretrain':
    enc_ctc_optimizer = optim.RMSprop(encoder_ctc.parameters(), lr=opt.lr)
    dec_ctc_optimizer = optim.RMSprop(decoder_ctc.parameters(), lr=opt.lr)
    dec_att_optimizer = optim.SGD(decoder_att.parameters(), lr=opt.lr)

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


def testCTCPretrain(encoder_ctc, decoder_ctc, dataset, criterion, n_aug=1):
    print('Start test set predictions')

    for p in encoder_ctc.parameters():
        p.requires_grad = False
    encoder_ctc.eval()

    for p in decoder_ctc.parameters():
        p.requires_grad = False
    decoder_ctc.eval()

    all_file_names = []
    all_preds = []
    image_count = 0
    pred_dict = {}

    for epoch in range(n_aug):
        test_iter = iter(dataset)
        for i in range(len(dataset)):
            data = test_iter.next()
            # i += 1
            cpu_images, __, file_names = data
            batch_size = cpu_images.size(0)
            image_count = image_count + batch_size
            utils.loadData(image, cpu_images)

            encoder_out = encoder_ctc(image)
            preds = decoder_ctc(encoder_out)
            # print(preds.size())
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

            # RA: While I am not sure yet, it looks like a greedy decoder and not beam search is being used here
            # Case is ignored in the accuracy, which is not ideal for an actual working system

            _, preds = preds.max(2)
            if torch.__version__ < '0.2':
                preds = preds.squeeze(2)  # https://github.com/meijieru/crnn.pytorch/issues/31
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

def testAttentionCTC(encoder_ctc, decoder_att, decoder_ctc, dataset, criterion, n_aug=1):
    print('Start test set predictions')

    for p in encoder_ctc.parameters():
        p.requires_grad = False
    encoder_ctc.eval()

    for p in decoder_ctc.parameters():
        p.requires_grad = False
    decoder_ctc.eval()

    all_file_names = []
    all_preds = []
    image_count = 0
    pred_dict = {}

    for epoch in range(n_aug):
        test_iter = iter(dataset)
        for i in range(len(dataset)):
            data = test_iter.next()
            # i += 1
            cpu_images, __, file_names = data
            batch_size = cpu_images.size(0)
            image_count = image_count + batch_size
            utils.loadData(image, cpu_images)

            encoder_out = encoder_ctc(image)
            preds = decoder_ctc(encoder_out)
            # print(preds.size())
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

            # RA: While I am not sure yet, it looks like a greedy decoder and not beam search is being used here
            # Case is ignored in the accuracy, which is not ideal for an actual working system

            _, preds = preds.max(2)
            if torch.__version__ < '0.2':
                preds = preds.squeeze(2)  # https://github.com/meijieru/crnn.pytorch/issues/31
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

def trainAttention( train_iter, enc, dec, encoder_optimizer, decoder_optimizer, criterion, max_length=models.crnn.MAX_LENGTH):


    data = train_iter.next()
    cpu_images, cpu_texts,__ = data
    # batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    target, target_length = converter.encode(cpu_texts)
    utils.loadData(text, target)
    utils.loadData(length, target_length)

    encoder_hidden = enc.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs = Variable(torch.zeros(max_length, 512)) # This 512 might have to change when the max-pool is changed.
    encoder_outputs = encoder_outputs.cuda() if opt.cuda else encoder_outputs

    loss = 0
    encoder_output = enc(image)

    target_variable = Variable(torch.LongTensor(target.cpu().numpy()).view(-1, 1)) #This is a hack. maybe there is a better way...
    target_variable = target_variable.cuda() if opt.cuda else target_variable

    input_length = len(encoder_output)

    for ei in range(input_length):
        encoder_outputs[ei] = encoder_output[ei][0]

    decoder_input = Variable(torch.LongTensor([[utils.SOS_token]]))
    decoder_input = decoder_input.cuda() if opt.cuda else decoder_input
    decoder_hidden = encoder_hidden

    # Teacher forcing: Feed the target as the next input
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = dec(
            decoder_input, decoder_hidden, encoder_outputs)

        loss += criterion(decoder_output, target_variable[di])
        decoder_input = target_variable[di]  # Teacher forcing


    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length.float()

def trainAttentionCTC(encoder_ctc,
                                  decoder_att,decoder_ctc, enc_ctc_optimizer, dec_att_optimizer, dec_ctc_optimizer, criterion_att,criterion_ctc,max_length=models.crnn.MAX_LENGTH):

    data = train_iter.next()
    cpu_images, cpu_texts,__ = data
    utils.loadData(image, cpu_images)
    target, target_length = converter.encode(cpu_texts)
    utils.loadData(text, target)
    utils.loadData(length, target_length)

    encoder_hidden = encoder_ctc.initHidden()

    dec_att_optimizer.zero_grad()

    loss = 0
    encoder_ctc_out = encoder_ctc(image)
    input_length = len(encoder_ctc_out)

    if input_length > models.crnn.MAX_LENGTH:
        print(input_length)
        print("Need to increase MAX_LENGTH to at least: "+str(input_length))

    encoder_outputs = Variable(torch.zeros(max_length, 512))
    encoder_outputs = encoder_outputs.cuda() if opt.cuda else encoder_outputs

    target_variable = Variable(torch.LongTensor(target.cpu().numpy()).view(-1, 1)) #This is a hack. maybe there is a better way...
    target_variable = target_variable.cuda() if opt.cuda else target_variable

    sample_idx =0
    if opt.batchSize>1:
        encoder_output = encoder_ctc_out[:,sample_idx,:] #grab first image

        for ei in range(input_length):
            encoder_outputs[ei] = encoder_output[ei]

        target_att_length = target_length[0]

    else:

        encoder_output = encoder_ctc_out
        input_length = len(encoder_output)
        for ei in range(input_length):
            encoder_outputs[ei] = encoder_output[ei, 0]

    decoder_input = Variable(torch.LongTensor([[utils.SOS_token]]))
    decoder_input = decoder_input.cuda() if opt.cuda else decoder_input
    decoder_hidden = encoder_hidden

    # Teacher forcing: Feed the target as the next input
    for di in range(target_att_length):
        decoder_output, decoder_hidden, decoder_attention = decoder_att(
            decoder_input, decoder_hidden, encoder_outputs)

        loss += criterion_att(decoder_output, target_variable[di])
        decoder_input = target_variable[di]  # Teacher forcing


    # att_cost = loss.data[0] / target_length[sample_idx]
    if opt.cuda:
        att_cost = loss / target_length[sample_idx]
    else:
        att_cost = torch.Tensor([loss / (target_length[sample_idx]).type(torch.FloatTensor)])
###CTC
    batch_size = cpu_images.size(0)
    decoder_output = decoder_ctc(encoder_ctc_out)
    preds = decoder_output
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion_ctc(preds, text, preds_size, length) / batch_size
    encoder_ctc.zero_grad()
    decoder_ctc.zero_grad()

    if opt.cuda:
        ctc_cost = cost.cuda()  #For some reason cost is on CPU and has to be explicitly specified on cuda before adding it with the other cost
    else:
        ctc_cost = cost

    if opt.mtlm:

        y_predict = mtlm(att_cost, ctc_cost)
        total_loss = criterion_mtlm(y_predict,torch.zeros(1))
        mtlm.zero_grad()
        # target_loss= torch.zeros(1)
        # total_loss = criterion_mtlm(out_loss, target_loss)
    else:

        alpha = .2
        total_loss = (1-alpha)*torch.log(att_cost) + alpha*torch.log(ctc_cost)

    total_loss.backward() # Note : We need to calculate the step size before we step

    enc_ctc_optimizer.step()
    dec_att_optimizer.step()
    dec_ctc_optimizer.step()

    if opt.mtlm:
        mtlm_optimizer.step()
        # print('mtlm under construction')
        return total_loss[0]
    else:
        return total_loss




def trainCTCPretrain(encoder_ctc,decoder_ctc, criterion, enc_ctc_optimizer,dec_ctc_optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts, __ = data

    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    encoder_out = encoder_ctc(image)
    preds = decoder_ctc(encoder_out)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    encoder_ctc.zero_grad()
    decoder_ctc.zero_grad()
    cost.backward()
    enc_ctc_optimizer.step()
    dec_ctc_optimizer.step()
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

def evaluateAtt(encoder_att, decoder_att, data, max_iter=1000):
    max_length = models.crnn.MAX_LENGTH

    # val_iter = iter(dataset)



    sim_preds = []
    raw_preds = []
    # max_iter = min(max_iter, len(dataset))
    gts = []
    # for i in range(max_iter):
        # data = val_iter.next()
        # i += 1
    cpu_images, cpu_texts, __ = data
    gts.append(cpu_texts[0])
    # target, target_length = converter.encode(cpu_texts)
    batch_size = cpu_images.size(0)
    # image_count = image_count + batch_size
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    encoder_hidden = encoder_att.initHidden()
    encoder_output = encoder_att(image)

    encoder_outputs = Variable(torch.zeros(max_length, 512))
    encoder_outputs = encoder_outputs.cuda() if opt.cuda else encoder_outputs

    input_length = len(encoder_output)

    for ei in range(input_length):
        encoder_outputs[ei] = encoder_output[ei][0]

    decoder_input = Variable(torch.LongTensor([[utils.SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if opt.cuda else decoder_input

    decoder_hidden = encoder_hidden
    decoder_attentions = torch.zeros(max_length, max_length)

    decoded_words = []
    pred_chars = []
    pred_chars_size = 0

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder_att(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        if ni == utils.EOS_token:
            # decoded_words.append('<EOS>') # This line is for debugging purposes. It is better to remove it for metrics
            break
        else:

            pred_chars.append(ni)
            pred_chars_size += 1

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if opt.cuda else decoder_input

    # sim_preds = decoded_words
    pc = torch.IntTensor(np.array(pred_chars))
    pcs = torch.IntTensor(np.array([pred_chars_size]))

    sim_pred = converter.decode(pc, pcs, raw=False)
    raw_pred = converter.decode(pc, pcs, raw=True)

    sim_preds.append(sim_pred)
    raw_preds.append(raw_pred)

    # return char_mean_error, word_mean_error, accuracy

    return sim_preds, cpu_texts, decoder_attentions[:di + 1], cpu_images[0,0]

def evaluateRandomly(enc, dec,test_loader,criterion, n=10):
    from matplotlib import pyplot as plt
    val_iter = iter(test_loader)
    for i in range(n):
        data = val_iter.next()
        output_sentence, target, attentions,image = evaluateAtt(enc, dec, data)

        fig = plt.figure()
        plt.subplot(211)
        plt.imshow(image)
        plt.subplot(212)
        plt.matshow(attentions.numpy())
        plt.savefig('results/'+str(i)+'.png')
        print('{0}<{1}'.format(target, output_sentence))
        print('')

def valAttention(encoder_att, decoder_att,dataset,criterion, max_iter=1000):

    print('Start validation set')

    max_length = models.crnn.MAX_LENGTH

    val_iter = iter(dataset)

    n_correct = 0
    loss_avg = utils.averager()

    image_count = 0
    # Character and word error rate lists
    char_error = []
    w_error = []

    max_iter = min(max_iter, len(dataset))

    sim_preds = []
    raw_preds = []
    max_iter = min(max_iter, len(dataset))
    gts = []
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts, __ = data
        gts.append(cpu_texts[0])
        target, target_length = converter.encode(cpu_texts)
        batch_size = cpu_images.size(0)
        image_count = image_count + batch_size
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        encoder_hidden = encoder_att.initHidden()
        encoder_output = encoder_att(image)

        encoder_outputs = Variable(torch.zeros(max_length, 512))
        encoder_outputs = encoder_outputs.cuda() if opt.cuda else encoder_outputs

        input_length = len(encoder_output)

        for ei in range(input_length):
            encoder_outputs[ei] = encoder_output[ei][0]

        decoder_input = Variable(torch.LongTensor([[utils.SOS_token]]))  # SOS
        decoder_input = decoder_input.cuda() if opt.cuda else decoder_input

        decoder_hidden = encoder_hidden
        decoder_attentions = torch.zeros(max_length, max_length)

        decoded_words = []
        pred_chars = []
        pred_chars_size = 0

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder_att(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            if ni == utils.EOS_token:
                # decoded_words.append('<EOS>') # This line is for debugging purposes. It is better to remove it for metrics
                break
            else:

                pred_chars.append(ni)
                pred_chars_size += 1

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if opt.cuda else decoder_input

        # sim_preds = decoded_words
        pc = torch.IntTensor(np.array(pred_chars))
        pcs = torch.IntTensor(np.array([pred_chars_size]))

        sim_pred = converter.decode(pc, pcs, raw=False)
        raw_pred = converter.decode(pc, pcs, raw=True)

        sim_preds.append(sim_pred)
        raw_preds.append(raw_pred)

    for pred, target in zip(sim_preds, gts):
        if pred == target:
            n_correct += 1

        # Case-insensitive character and word error rates
        char_error.append(cer(pred, target))
        w_error.append(wer(pred, target))

    for raw_pred, pred, gt in zip(raw_preds[:opt.n_test_disp], sim_preds, gts):
        print('%-20s => %-20smmm, gt: %-20s' % (raw_pred, pred, gt))

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

def valAttentionCTC(encoder_ctc, decoder_att, decoder_ctc, dataset, criterion, max_iter=1000):

    print('Start validation set')

    ####### CTC ######
    for p in encoder_ctc.parameters():
        p.requires_grad = False

    for p in decoder_ctc.parameters():
        p.requires_grad = False

    encoder_ctc.eval()
    # dec_att.eval()
    decoder_ctc.eval()


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

        enc_output = encoder_ctc(image)
        preds = decoder_ctc(enc_output)
        # print(preds.size())
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        if torch.__version__ < '0.2':
            preds = preds.squeeze(2)  # https://github.com/meijieru/crnn.pytorch/issues/31
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
        print('CTC==>%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    ##### Attention ####


    max_length = models.crnn.MAX_LENGTH

    val_iter = iter(dataset)

    n_correct = 0
    #loss_avg = utils.averager()

    image_count = 0


    sim_preds = []
    raw_preds = []
    max_iter = min(max_iter, len(dataset))
    gts = []
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts, __ = data
        gts.append(cpu_texts[0])
        target, target_length = converter.encode(cpu_texts)
        batch_size = cpu_images.size(0)
        image_count = image_count + batch_size
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        encoder_hidden = encoder_ctc.initHidden()
        encoder_output = encoder_ctc(image)

        encoder_outputs = Variable(torch.zeros(max_length, 512))
        encoder_outputs = encoder_outputs.cuda() if opt.cuda else encoder_outputs

        input_length = len(encoder_output)

        for ei in range(input_length):
            encoder_outputs[ei] = encoder_output[ei][0]

        decoder_input = Variable(torch.LongTensor([[utils.SOS_token]]))  # SOS
        decoder_input = decoder_input.cuda() if opt.cuda else decoder_input

        decoder_hidden = encoder_hidden
        decoder_attentions = torch.zeros(max_length, max_length)

        decoded_words = []
        pred_chars = []
        pred_chars_size = 0

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder_att(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            if ni == utils.EOS_token:
                # decoded_words.append('<EOS>') # This line is for debugging purposes. It is better to remove it for metrics
                pred_chars.append(ni)
                pred_chars_size += 1
                break
            else:

                pred_chars.append(ni)
                pred_chars_size += 1

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if opt.cuda else decoder_input

        # sim_preds = decoded_words
        pc = torch.IntTensor(np.array(pred_chars))
        pcs = torch.IntTensor(np.array([pred_chars_size]))

        sim_pred = converter.decode(pc, pcs, raw=False)
        raw_pred = converter.decode(pc, pcs, raw=True)

        sim_preds.append(sim_pred)
        raw_preds.append(raw_pred)

    for raw_pred, pred, gt in zip(raw_preds[:opt.n_test_disp], sim_preds, gts):
        print('Attention==>%-20s => %-20smmm, gt: %-20s' % (raw_pred, pred, gt))

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

def valCTCPretrain(encoder_ctc, decoder_ctc, dataset, criterion, max_iter=1000):
    print('Start validation set')

    for p in encoder_ctc.parameters():
        p.requires_grad = False
    encoder_ctc.eval()

    for p in decoder_ctc.parameters():
        p.requires_grad = False
    decoder_ctc.eval()

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

        encoder_out = encoder_ctc(image)
        preds = decoder_ctc(encoder_out)

        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        # RA: While I am not sure yet, it looks like a greedy decoder and not beam search is being used here
        # Case is ignored in the accuracy, which is not ideal for an actual working system

        _, preds = preds.max(2)
        if torch.__version__ < '0.2':
            preds = preds.squeeze(2)  # https://github.com/meijieru/crnn.pytorch/issues/31
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

            if opt.model=='attention':
                loss = trainAttention(train_iter, encoder,
                             attn_decoder, encoder_optimizer, decoder_optimizer, criterion)
            elif opt.model=='ctc':
                for p in crnn.parameters():
                    p.requires_grad = True
                crnn.train()

                loss = trainBatch(crnn, criterion, optimizer) # it trains/backpropagates once/batch, each batch is made up of "batchSize" images
            elif opt.model=='attention+ctc':

                setupTrain(encoder_ctc)
                setupTrain(decoder_ctc)

                if opt.mtlm:
                    setupTrain(mtlm)

                loss = trainAttentionCTC( encoder_ctc,
                                      decoder_att,decoder_ctc, enc_ctc_optimizer, dec_att_optimizer, dec_ctc_optimizer, criterion_att,criterion_ctc)
            elif opt.model=='ctc_pretrain':

                setupTrain(encoder_ctc)
                setupTrain(decoder_ctc)

                loss = trainCTCPretrain(encoder_ctc,decoder_ctc, criterion, enc_ctc_optimizer, dec_ctc_optimizer)

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
                if opt.model=='attention':
                    val_CER, val_WER, val_ACC = valAttention(encoder,attn_decoder, test_loader, criterion)
                    train_CER, train_WER, train_ACC = valAttention(encoder, attn_decoder, train_loader, criterion)
                elif opt.model=='ctc':
                    val_CER, val_WER, val_ACC = val(crnn, test_loader, criterion)
                    train_CER, train_WER, train_ACC = val(crnn, train_loader, criterion)
                elif opt.model=='attention+ctc':
                    val_CER, val_WER, val_ACC = valAttentionCTC(encoder_ctc,decoder_att,decoder_ctc, test_loader, criterion_ctc)
                    train_CER, train_WER, train_ACC = valAttentionCTC(encoder_ctc, decoder_att, decoder_ctc, train_loader,
                                                                criterion_ctc)
                elif opt.model=='ctc_pretrain':
                    val_CER, val_WER, val_ACC = valCTCPretrain(encoder_ctc,decoder_ctc, test_loader, criterion)
                    train_CER, train_WER, train_ACC = valCTCPretrain(encoder_ctc, decoder_ctc, train_loader, criterion)

                history_errors.append([epoch, i, loss,train_ACC,train_WER,train_CER,val_ACC,val_WER,val_CER])

                if opt.plot:
                    utils.savePlot(history_errors,model_rpath)

            # do checkpointing
            if (epoch % opt.saveEpoch == 0) and (i >= len(train_loader)):      # Runs at end of some epochs
                print("Saving epoch",  '{0}/netCRNN_{1}_{2}.pth'.format(model_rpath, epoch, i))

                if opt.model=='attention':
                    torch.save(encoder.state_dict(), '{0}/netCNN_{1}_{2}.pth'.format(model_rpath, epoch, i))
                    torch.save(attn_decoder.state_dict(), '{0}/netAttnDec_{1}_{2}.pth'.format(model_rpath, epoch, i))
                elif opt.model=='ctc':
                    torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(model_rpath, epoch, i))
                elif opt.model=='attention+ctc':
                    torch.save(encoder_ctc.state_dict(), '{0}/netCNN_{1}_{2}.pth'.format(model_rpath, epoch, i))
                    torch.save(decoder_att.state_dict(), '{0}/netAttnDec_{1}_{2}.pth'.format(model_rpath, epoch, i))
                    torch.save(decoder_ctc.state_dict(), '{0}/netCTCDec_{1}_{2}.pth'.format(model_rpath, epoch, i))
                elif opt.model=='ctc_pretrain':
                    torch.save(encoder_ctc.state_dict(), '{0}/netCNN_{1}_{2}.pth'.format(model_rpath, epoch, i))
                    torch.save(decoder_ctc.state_dict(), '{0}/netCTCDec_{1}_{2}.pth'.format(model_rpath, epoch, i))

elif opt.mode=='test':
    if opt.dataset=='ICFHR':
        if opt.model=='ctc':
            files, predictions = test(crnn, test_loader, criterion, n_aug=opt.n_aug if opt.test_aug else 1)
        elif opt.model=='ctc_pretrain':
            files, predictions = testCTCPretrain(encoder_ctc,decoder_ctc, test_loader, criterion, n_aug=opt.n_aug if opt.test_aug else 1)
        elif opt.model=='attention+ctc':
            files, predictions = testAttentionCTC(encoder_ctc,decoder_att,decoder_ctc, test_loader, criterion_ctc, n_aug=opt.n_aug if opt.test_aug else 1)
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

elif opt.mode=='random-test':
    evaluateRandomly(encoder,attn_decoder,test_loader,criterion)





