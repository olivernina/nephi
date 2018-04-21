#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image, ImageOps
#import cv2
import numpy as np
from grid_distortion import warp_image
encoding = 'utf-8'

# This link has good tutorial for understanding data loaders for pytorch.
# https://www.reddit.com/r/MachineLearning/comments/6ado09/p_a_comprehensive_tutorial_for_image_transforms/

# Curtis Wigington's code also shows how to do this
#https://github.com/cwig/simple_hwr/blob/master/hw_dataset.py

# RA April 2018: Another idea I have is to shrink and expand images randomly on individual call. This will allow learning features at different scales of the training set is not sufficiently variable in size/resolution.

# RA 10 April 2018: With the modifications I did today to include binarized images, the code doesn't throw errors. It remains to be seen if training results in better validation error.

# RA 12 April 2018: With what I learned today about dataloaders in pytorch, I think I should stop calling randomsequential sampler and instead make shuffle=True so that images are dynamically loaded and batches dynamically made. I should make a class for the grid_distortion as a transform. Or as Curtis does, I can just add it in as a True/False parameter for now. Curtis has shuffle as False, but he also fixes the aspect ratio for everything. I guess doing what he does could help determine how to do things (perform well on the German data). When they fix the aspect ratio, then that gets everything. But what about in batches? Yes, they just fill in to the max size, though maybe in a different way than I do. It's about the same. It probably shouldn't matter. But I could go to his way if need be.

# - I should also possibly take a small sample of images and practice loading them in and seeing what happens at different epochs, with the dataloader, but I think I know how it works now.

class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None, binarize=False):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'))
            self.nSamples = 100#nSamples
        
        self.binarize = binarize
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            if self.binarize:
                howe_imageKey = 'howe-image-%09d' % index
                simplebin_imageKey = 'simplebin-image-%09d' % index
            
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            if self.binarize:
                imgbuf = txn.get(howe_imageKey)
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                try:
                    img_howe = Image.open(buf).convert('L')
                except IOError:
                    print('Corrupted image for %d' % index)
                    return self[index + 1]
                imgbuf = txn.get(simplebin_imageKey)
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                try:
                    img_simplebin = Image.open(buf).convert('L')
                except IOError:
                    print('Corrupted image for %d' % index)
                    return self[index + 1]

            # When I do a transformation here, I should first combine the images into one.
            
            #if self.transform is not None:
            #    img = self.transform(img)
            #    if self.binarize:
            #        img_howe = self.transform(img_howe)
            #        img_simplebin = self.transform(img_simplebin)
            
            # NO, since I use Image package in the dataset part, then I have to change everything. I think I should just change it in the warp_image code to cv2, then pack to Pillow, or switch to Pillow here. numpy.array is the basic intermediary
            # basically, I just use numpy array to change things, the grid doesn't care how colors are affected, I think.
            #opencvImage = cv2.cvtColor(numpy.array(PILImage), cv2.COLOR_RGB2BGR)
            #img = Image.fromarray(warp_image(np.array(img), w_mesh_interval=100, h_mesh_interval=40, w_mesh_std=10, h_mesh_std=4))
            
            # Now I have to figure out how to combine the images...
            
            label_key = 'label-%09d' % index
            label = unicode(txn.get(label_key), encoding=encoding)   # Hopefully this still works with unicode
            
            # I want the other thing not to have a problem with lmdb databases already here, okay, .get() should return None if there is nothing
            file_key = 'file-%09d' % index
            file_name = str(txn.get(file_key)) 

            if self.target_transform is not None:
                label = self.target_transform(label)
            
            final_image = Image.merge("RGB", (img, img_howe, img_simplebin)) if self.binarize else img
            if self.transform is not None:
                final_image = self.transform(final_image)

            DEBUG = False
            if DEBUG:
                print("The image has shape:")
                print(np.array(final_image).shape)
       
            return (final_image, label, file_name)

    # Hopefully the merging will work with the batch functions.

# RA: I think more work for making sure this code functions when the image height is larger than the resize height. However, when I ran the main module with this resizeNormalize function, i got no errors

# RA 5 Mar 2018: I think for a transfer learning approach, we should not resize the images at all. We should just feed in the largest image height and width (independently) and pad the remaining space.




class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        
        # Resize image as necessary to new height, maintaining aspect ratio
        o_size = img.size
        AR = o_size[0] / float(o_size[1])
        img = img.resize((int(round(AR * self.size[1])), self.size[1]), self.interpolation)
        
        # Now pad to new width, as target width is guaranteed to be larger than width if keep aspect ratio is true
        o_size = img.size
        delta_w = self.size[0] - o_size[0]
        delta_h = self.size[1] - o_size[1]
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        new_im = ImageOps.expand(img, padding, "white")
        
        img = self.toTensor(new_im)
        img.sub_(0.5).div_(0.5)
        
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=80, imgW=300, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels, files = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size  # PIL doesn't return channel number so w, h is all
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            
            #RA: I don't understand the purpose of this line, and for handwriting recognition imgW >= imgH
            #imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)   # Make sure this performs correctly with 3 channel images, it should, as it is adding a dimension for the batch and putting all the images together

        return images, labels, files
