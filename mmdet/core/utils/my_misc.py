# my_misc.py qqq
import torch
import numpy as np
import datetime
import pickle
import cv2

import matplotlib
#matplotlib.use('WebAgg')
matplotlib.use('Agg') #otherwise we may get ## matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
from numpy import unravel_index
from matplotlib.backends.backend_pdf import PdfPages
import os

from mmcv.runner import HOOKS, Hook

class struct():
    pass


def t2i(t):
    # swap color axis because
    # torch image: C x H x W
    # numpy image: H x W x C
    im = t.detach().numpy().astype('uint8').transpose((1, 2, 0))
    return im
#def t2i(im):

def t2i_a(t):
    # swap color axis because
    # torch image: C x H x W
    # numpy image: H x W x C
    im = t.detach().numpy().transpose((1, 2, 0))
    return im
#def t2i_a(im):

def i2t(im):
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C x H x W
    im2 = im.astype('float32').transpose((2, 0, 1))
    t = torch.from_numpy(im2)
    return t
#def i2t(im):

def im2ImDisp2(im, norm=0,convert_gray_to_rgb=0):
    im = im.copy()
    # construct imDisp: 'saved' image for ndisplay, where from the image copies are made for dispalyin in different axws
    assert type(im).__module__ == np.__name__  # is an np opject
    if im.dtype.name == 'float32':
        if norm == 1:
            imDisp = ((im - np.min(im)) / (np.max(im) - np.min(im)) * 255).astype('uint8')
        else:
            imDisp = np.uint8(im)
    elif im.dtype.name == 'bool':
        imDisp = im.astype('float32')
    elif im.dtype.name == 'uint8':
        imDisp = im.copy()
    else:
        print('non supported im.dtype.name=' + im.dtype.name)
        imDisp = 0

    if convert_gray_to_rgb:
        if len(imDisp.shape) == 2:

            imDisp = np.stack((imDisp, imDisp, imDisp), axis=2)


    return imDisp


# def im2ImDisp(im):


def d0a(im,suptitle='',pars=[], gray = 0, matshow = 0, save_to_and_close_the_pdf = 0, di_sav_dbg = [], pdf_prefix = '', fn = []):
#save_pdf for debugging purposes

    matplotlib.use('TKAgg')

    assert not (gray and matshow )

    if save_to_and_close_the_pdf:
        assert di_sav_dbg != []

    fig = plt.figure(figsize=(5, 5))
    plt.suptitle(suptitle)

    if matshow:
        plt.matshow(im,fignum=False)
    elif gray or len(im.shape)==2:
        plt.imshow(im,cmap=cm.gray)
    else:
        plt.imshow(im)

    if save_to_and_close_the_pdf:

        if fn:
            fn1 = di_sav_dbg + '/' + fn + '.pdf'
        else:
            #'17.09.20,10:07:22'
            stri=datetime.datetime.now().strftime('%d.%m.%y,%H:%M:%S.%f')
            stri = 'tmp_' + stri
            stri = stri.replace(',','_')
            stri = stri.replace('.','_')
            stri = stri.replace(':','h_',1)
            stri = stri.replace(':','min_',1)
            stri = stri.replace(':','sec_',1)
            stri += 'msec'
            fn1 = di_sav_dbg + '/' + pdf_prefix + stri+ '_out.pdf'
        pdf_pages_loc = PdfPages(fn1)
        print('pdf saved in ' + fn1)

        pdf_pages_loc.savefig()
        plt.close(fig)

        pdf_pages_loc.close()

    elif pars!=[] and pars['pdf']:

        pars['PdfPages'].savefig()
        plt.close(fig)

    else:
        plt.pause(0.001);
        plt.imshow(im)

