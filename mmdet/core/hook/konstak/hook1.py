#hook1.py

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class hook1(Hook):

    def __init__(self, a, b):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        print('we are in before_iter()')
        pass

    def after_iter(self, runner):
        print('we are after before_iter()')
        pass
