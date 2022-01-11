#my_hook.py qqq

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

#from mmdet.core.utils.my_misc import struct, t2i_a, d0a
from mmdet.core.utils.my_misc import d0a

@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, a, b):
        self.n_iter = 0
        pass

    def before_run(self, runner):
        print("we are before_run")
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        #runner.model.module.backbone.conv1.weight[10].cpu().numpy().shape
        #(3, 7, 7)




        if 0:
            #https://discuss.pytorch.org/t/accessing-layers-inside-bottleneck-module-of-pretrained-resnet-model/16287
            nb=0
            for ch in runner.model.module.backbone.layer4.children():
                print('--------------- nb='+str(nb))
                print (ch)
                nb += 1
                if nb==2:
                    ch.conv2.weight.shape
                    #   Out[35]: torch.Size([512, 512, 3, 3])

        if 0:
            n_flt = 10
            flt = t2i_a(runner.model.module.backbone.conv1.weight[n_flt].cpu())
            d0a(flt,save_to_and_close_the_pdf = 1, di_sav_dbg = '/home/konstak/projects2/mmdetection/demo', \
                    pdf_prefix = '', fn = 'n_flt=' + str(n_flt) + '_before_n_iter_' + str(self.n_iter))

        # print('we are in before_iter()')
        pass

    def after_iter(self, runner):
        self.n_iter += 1
        # print('we are after before_iter()')
        pass
