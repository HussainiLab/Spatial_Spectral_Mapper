# -*- coding: utf-8 -*-
"""
Created on Mon Oct  11 12:42:14 2021

@author: vajramsrujan
"""

from PyQt5.QtCore import QThread
from .WorkerSignals import WorkerSignals

# =========================================================================== #

class Worker(QThread): 
    
    def __init__(self, function, *args, **kwargs):
        
        QThread.__init__(self)
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.data = None
        
        self.signals = WorkerSignals()
        
    # ------------------------------------------- #   
    
    def run(self, **kwargs):
        self.data = self.function(self, *self.args, **self.kwargs)
        self.signals.image_data.emit( (self.data[0], self.data[1], self.data[2], self.data[3], self.data[4], self.data[5]) )

# =========================================================================== #