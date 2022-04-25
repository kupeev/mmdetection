# qq
# general_dk
# from demo.davidk.general_dk import *

import os
from matplotlib.widgets import Button
import builtins
import shutil

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

import numpy as np
import datetime
import pickle
import cv2

import matplotlib
#matplotlib.use('WebAgg')
#matplotlib.use('Agg') #otherwise we may get matplotlib.use('Agg')
#matplotlib.use('Agg') #otherwise we may get matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
from numpy import unravel_index
from matplotlib.backends.backend_pdf import PdfPages
import os

class struct():
    pass

eps = np.finfo(np.float32).eps


def find_t(val, lst):
    #find(5, [1, 2, 10]) == []
    #find(10, [1, 2, 10,10]) == [2,3]
    """
    lst = []
    lst.append(1)
    lst.append(2)
    lst.append(10)
    val = 3
    """
    ii = []
    for (i, val1) in enumerate(lst):
        try:
            if val1 == val:
                ii.append(i)
        except:
            tmp=10
    return ii

def embedRect(im, tlx, tly, brx, bry, meth='const', val=0, randrange=[220, 240], Type='uint8'):
    # fills rectangle in the im, im may be empty

    if type(im) == list and im == []:
        imhe = bry - tly + 1
        imwi = brx - tlx + 1
        if meth == 'rand':
            if Type == 'float32':
                im = randrange[0] + np.random.rand(imhe, imwi) * (randrange[1] - randrange[0])
                im = im.astype(np.float32)
            elif Type == 'uint8':
                im = np.random.randint(randrange[0], randrange[1], (imhe, imwi), Type)
        elif meth == 'const':
            im = np.zeros((imhe, imwi)).astype(Type)
            if Type == 'uint8':
                im.fill(np.uint8(val))
            else:
                assert False, 'non supported Type'
    else:
        pathe = bry - tly + 1
        patwi = brx - tlx + 1
        if meth == 'rand':
            pat = np.random.randint(220, 240, (pathe, patwi), Type)
        elif meth == 'const':
            pat = np.zeros((pathe, patwi)).astype(Type)
            if Type == 'uint8':
                pat.fill(np.uint8(val))
            else:
                assert False, 'non supported Type'

        im[tly:bry + 1, tlx:brx + 1] = pat

    return im

def im_2_imuint16(im):
    im1 = im.astype('float32')
    im1 = (im1 - np.min(im1)) / (np.max(im1) - np.min(im1)) * np.iinfo(np.uint16).max
    im1 = im1.astype('uint16')
    return im1

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


def d0a(im,suptitle='',pars=[], gray = 0, matshow = 0, save_to_and_close_the_pdf = 0, \
        di_sav_dbg = [], pdf_prefix = '', add_time = 0):
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

        if add_time:
            #'17.09.20,10:07:22'
            stri=datetime.datetime.now().strftime('%d.%m.%y,%H:%M:%S.%f')
            stri = 'tmp_' + stri
            stri = stri.replace(',','_')
            stri = stri.replace('.','_')
            stri = stri.replace(':','h_',1)
            stri = stri.replace(':','min_',1)
            stri = stri.replace(':','sec_',1)
            stri += 'msec'
        else:
            stri = ''
        fn = di_sav_dbg + '/' + pdf_prefix + stri+ '_out.pdf'
        pdf_pages_loc = PdfPages(fn)
        print('pdf saved in ' + fn)

        pdf_pages_loc.savefig()
        plt.close(fig)

        pdf_pages_loc.close()

    elif pars!=[] and pars['pdf']:

        pars['PdfPages'].savefig()
        plt.close(fig)

    else:
        plt.pause(0.001);
        plt.imshow(im)

def d0g(im):
    plt.imshow(im, cmap='gray');plt.show()

def dd(im):
# uint8, 3-d
    plt.figure();
    plt.imshow(im);
    plt.pause(0.001);
    plt.show()

#uint8,

def argmax2(array2d):
    return unravel_index(array2d.argmax(), array2d.shape)

def str_2_nparray(stri, pars = []):

    max_len = pars['metadata_string_length'] if pars != [] else 1000

    #nparray = np.zeros((nExamples, 500)).astype('int8')
    nparray = np.zeros((max_len, )).astype('int8')

    assert len(stri) + 1 <= max_len
    for i in range(len(stri)):
        nparray[i] = ord(stri[i])
    nparray[i + 1] = -1

    if 1:
        stri_2 = npvector_2_str(nparray)
        assert stri == stri_2, 'probably metadata_string_length is too small'

    return nparray

def npvector_2_str(npvector0):
# example: npvector0 of (1000,)
# returns stri: '{"fn": "00010764.tif", "id": 36, .............  "patch_sz": 43}'

    npvector = npvector0.copy()
    ind = np.nonzero(npvector == -1)[0][0]
    npvector = npvector[0:ind]
    npvector1 = npvector.astype('uint8')
    stri = npvector1.tostring()
    stri = stri.decode('utf-8')
    return stri
#def npvector_2_str(npvector0):


def nparray_2_listofstrings(nparray):
#example: nparray.shape  (8, 1000)
#returns list of 8 strings
    lst = []
    for i in range(nparray.shape[0]):
        vect = nparray[i,:]
        stri = npvector_2_str(vect)
        lst.append((stri))
    return lst
#def numpyarray_2_str(nparray):


def plot(pars,y_list, fmt_list, label_list, title = '', xlabel = '', ylabel = '', ylim = [] ):
    fig = plt.figure(figsize=(5, 5))

    assert len(y_list) == len (fmt_list) and len (fmt_list)  == len(label_list)

    for i in range( len(y_list) ):
        plt.plot(range(len(y_list[i])), y_list[i], fmt_list[i], label=label_list[i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    #for i in range(len(y_list)):

    plt.title(title)

    if ylim == []:
        ylim_new = [np.min(y_list[i]), np.max(y_list[i])]
    elif ylim[0]==ylim[1]:
        if ylim[0]==0:
            ylim_new = [0, 0.001]
        else:
            ylim_new = [ylim[0], ylim[0]*0.1]
    else:
        ylim_new = ylim

    plt.ylim(ylim_new)

    plt.legend()

    if pars['pdf']:
        pars['PdfPages'].savefig()
        plt.close(fig)
        tmp=10

    else:
        plt.show()

    return

#def plot(pars, y_list, fmt_list, label_list, title='', xlabel='', ylabel='', ylim=[]):


def init_pdf(pars):

    pars['PdfPages'] = PdfPages(pars['dbg: pdf fn'])
    plt.rcParams["axes.titlesize"] = 7
    plt.rcParams["figure.titlesize"] = 7
    # plt.rcParams["axes.titlepad"] = 20
    cmap = plt.get_cmap('bwr')

def close_pdf(pars):
    print('pdf saved: ' + pars['dbg: pdf fn'])
    pars['PdfPages'].close()


def Print(C, precision=3):
    with np.printoptions(precision=precision, suppress =True):
        print (C)
#def Print(C, precision=3):

def remove_zero_cols_rows_2d(m0, val = 0):
#m is a matrix, removed arr cos and row completely consistiong of val values

    assert False, 'TBD: to test the function, follwing remove_zero_cols_rows_3d '

    m = m0.copy()

    m[m==val]=0

    tmp=10

    zerocols = np.argwhere(np.all( m==0 , axis=0  ))
    m1 = np.delete(m,zerocols, axis=1)

    zerorows = np.argwhere(np.all( m==0 , axis=1  ))
    m2 = np.delete(m1,zerorows, axis=0)

    return m2


def remove_zero_cols_rows_3d(m0_3d, val3d = [255,255,255]):
#input
#   m0_3d is he x wi x 3 image
#   the func removes from m0_3d the rows and cols completely consisting of val3d.
#
#   remark: usually m0_3d is an origibal semantic model on white background
#
#returns:
#   m0_3d after the removal,
#   i0, j0 are the starting cropping indexes in the initial matrix


    m_3d = m0_3d.copy()

    m1=(m_3d[:,:]==[val3d[0],val3d[1],val3d[2]])
    #m2: not (all m1 are true ie val3d is bkgr)
    # at least one m1 is false == at least val3d is not bkgr
    #m2: not (all m1 are true ie val3d is bkgr)
    #m2 = np.logical_and(m1[:,:,0],m1[:,:,1],m1[:,:,2])==False
    m2 = (  (m1[:,:,0] & m1[:,:,1] &  m1[:,:,2])  ) == False

    tmp=10

    """
    zerocols = np.argwhere(np.all( m2==False , axis=0  ))
    m3 = np.delete(m2,zerocols, axis=1)

    zerorows = np.argwhere(np.all( m2==False , axis=1  ))
    m4 = np.delete(m3,zerorows, axis=0)
    """
    zerocols = np.argwhere(np.all( m2==False , axis=0  ))
    m_3d_nocols = np.delete(m_3d,zerocols, axis=1)
    # zerocols = 0,1,2,....,100, 125,126,127
    j0 = [j for j in range(len(zerocols)) if zerocols[j]!=j] [0]
    #j0 is the first index zerocols[j0]!=u (101 in the above example)

    zerorows = np.argwhere(np.all( m2==False , axis=1  ))
    i0 = [i for i in range(len(zerorows)) if zerorows[i]!=i] [0]
    m_3d_nocols_norows = np.delete(m_3d_nocols,zerorows, axis=0)

    #i0, j0 are the starting cropping indexes in the initial matrix

    assert np.array_equal( m_3d[i0:i0+m_3d_nocols_norows.shape[0],  j0:j0+m_3d_nocols_norows.shape[1]  ] ,   m_3d_nocols_norows )

    return m_3d_nocols_norows, i0, j0

#def remove_zero_cols_rows_3d(m0_3d, val3d = [0,0,0]):


def pad_image(im, pad,zerofizepads=0):
# if im is a 3-d
#   pads 2-d scilces of an input 3-d image by the mean (excluding pads) of the slice
# if im is a 2-d
#   pads an input image by the mean (excluding pads)

    if len(im.shape) == 3:
        #im = np.zeros((imtf.shape[0] + hp + hp, imtf.shape[1] + hp + hp)).astype(imtf.dtype.name)

        means_last_dim = np.mean(im[pad:-pad,pad:-pad,:],axis=(0,1))
        im2 = np.zeros(im.shape)
        for k in range(im.shape[2]):
            if zerofizepads==0:
                im2[:,:,k].fill(means_last_dim[k])
            im2[pad:-pad,pad:-pad,k] = im[pad:-pad,pad:-pad,k]
    else:

        assert len(im.shape) == 2
        mean_im = np.mean(im[pad:-pad,pad:-pad])
        im2 = np.zeros_like(im)
        if zerofizepads==0:
            im2[:,:].fill(mean_im)
        im2[pad:-pad,pad:-pad] = im[pad:-pad,pad:-pad]

    return im2
#def pad_image(im, pad):

def dir_di(in_dir):
    #returns list all files of in_dir non-recurs
    fns2=[]
    for root, subdir, fns in os.walk(in_dir):
        if subdir==[]:
            for fn in fns:
                if fn=='.directory':
                    continue
                fns2.append(fn)
        else:
            tmp=10
    return fns2

#def load_db()


def i2d(im):
    im1 = np.float32(im.copy() )
    im1 = np.uint8((255*(im1-np.min(im1))/(np.max(im1)-np.min(im1))))
    return im1

def Str(val, prec = 2):
    if val == [] or (isinstance(val,np.ndarray) and len(val) == 0):
        stri = '[]'
    else:
        stri = str( round( val, prec ) )
    return stri
def get_run_id():

    stri_dt=datetime.datetime.now().strftime('%d.%m.%y,%H:%M:%S.%f')
    #'25.07.21,12:19:49.358241'
    stri_dt = stri_dt.replace(',','_')
    #'25.07.21_12:19:49.358241'
    inds = [i for i in range(len(stri_dt)) if stri_dt.startswith(':',i)]
    stri_dt2 = stri_dt[0:inds[-1]]
    stri_dt2 = stri_dt2.replace(':','.')
    li=list(stri_dt2)
    li[inds[-2]]='h'
    li.append('min')
    run_id="".join(li)
    return run_id #'26.07.21_12h51min'

#def get_run_id():

def is_empty(x):
    return np.array_equal(x,[])

def t2i(t):
    # torch image: C x H x W
    # numpy image: H x W x C
    #im = t.detach().numpy().astype('uint8').transpose((1, 2, 0))

    assert len(t.shape) == 3, 't2i(): dim error, probablyv should call t2i(im[0])'
    try:
        im = t.detach().cpu().numpy().transpose((1, 2, 0))
    except:
        tmp=10
    return im
#def t2i(im):

def i2t(im):
    # numpy image: H x W x C
    # torch image: C x H x W
    #im2 = im.astype('float32').transpose((2, 0, 1))
    im2 = im.transpose((2, 0, 1))
    t = torch.from_numpy(im2)
    return t
#def i2t(im):

if 0:


    sdjkcasd

