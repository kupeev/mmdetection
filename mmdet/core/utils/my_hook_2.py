#my_hook_2.py qqq

#for mesima_2

from mmdet.core.utils.my_misc import d0a

# follwing https://towardsdatascience.com/the-one-pytorch-trick-which-you-should-know-2d5e9c1da2ca

class SaveLayerOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        print('len(self.outputs)='+str(len(self.outputs)))
        self.outputs.append(module_out)

        """
        self.outputs[0][0][0][0][2].shape
        Out[17]: torch.Size([104, 336])
        self.outputs[0][0][0][0][0].shape
        Out[18]: torch.Size([104, 336])
        self.outputs[0][0][0][0][1].shape
        Out[19]: torch.Size([104, 336])
        """
        if len(self.outputs) == 23:
            # d0a(self.outputs[10].cpu().numpy()[0,63, :,:])
            tmp=10

    def clear(self):
        self.outputs = []