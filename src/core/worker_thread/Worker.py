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
        # Emit full data tuple to support optional binned_data (backward compatible in receiver)
        self.signals.image_data.emit(self.data)

# =========================================================================== #