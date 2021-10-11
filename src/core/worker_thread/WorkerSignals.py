# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 16:40:32 2021

@author: vajra
"""

from PyQt5.QtCore import QObject, pyqtSignal

# =========================================================================== #    

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    '''
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    text_progress = pyqtSignal(str)
    image_data = pyqtSignal(tuple)
    
# =========================================================================== # 