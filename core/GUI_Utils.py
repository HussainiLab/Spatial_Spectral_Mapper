import os
from PyQt4 import QtGui, QtCore
import matplotlib
matplotlib.use('QT4Agg')
from core.Tint_Matlab import *


def background(self):  # defines the background for each window
    """providing the background info for each window"""
    # Acquiring information about geometry
    self.PROJECT_DIR = os.path.dirname(os.path.abspath("__file__"))  # project directory
    self.IMG_DIR = os.path.join(self.PROJECT_DIR, 'img')  # image directory
    self.CORE_DIR = os.path.join(self.PROJECT_DIR, 'core')  # core directory
    self.SETTINGS_DIR = os.path.join(self.PROJECT_DIR, 'settings')  # settings directory
    if not os.path.exists(self.SETTINGS_DIR):
        os.mkdir(self.SETTINGS_DIR)

    self.setWindowIcon(QtGui.QIcon(os.path.join(self.IMG_DIR, 'cumc-crown.png')))  # declaring the icon image
    self.deskW, self.deskH = QtGui.QDesktopWidget().availableGeometry().getRect()[2:] #gets the window resolution
    # self.setWindowState(QtCore.Qt.WindowMaximized) # will maximize the GUI
    self.setGeometry(0, 0, self.deskW/2, self.deskH/1.5)  # Sets the window size, 800x460 is the size of our window

    QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('Cleanlooks'))


class Worker(QtCore.QObject):
    # def __init__(self, main_window, thread):
    def __init__(self, function, *args, **kwargs):
        super(Worker, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.start.connect(self.run)

    start = QtCore.pyqtSignal(str)

    @QtCore.pyqtSlot()
    def run(self):

        self.function(*self.args, **self.kwargs)


def center(self):
    """centers the window on the screen"""
    frameGm = self.frameGeometry()
    screen = QtGui.QApplication.desktop().screenNumber(QtGui.QApplication.desktop().cursor().pos())
    centerPoint = QtGui.QApplication.desktop().screenGeometry(screen).center()
    frameGm.moveCenter(centerPoint)
    self.move(frameGm.topLeft())


class Communicate(QtCore.QObject):
    '''A custom pyqtsignal so that errors and popups can be called from the threads
    to the main window'''
    myGUI_signal = QtCore.pyqtSignal(str)


def find_consec(data):
    '''finds the consecutive numbers and outputs as a list'''
    consecutive_values = []  # a list for the output
    current_consecutive = [data[0]]

    if len(data) == 1:
        return [[data[0]]]

    for index in range(1, len(data)):

        if data[index] == data[index - 1] + 1:
            current_consecutive.append(data[index])

            if index == len(data) - 1:
                consecutive_values.append(current_consecutive)

        else:
            consecutive_values.append(current_consecutive)
            current_consecutive = [data[index]]

            if index == len(data) - 1:
                consecutive_values.append(current_consecutive)
    return consecutive_values