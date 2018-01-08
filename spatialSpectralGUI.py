from PyQt4 import QtGui, QtCore
import sys, os, time, datetime, functools
from PIL import Image
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from core.spatialSpectralFunctions import *


class Worker(QtCore.QObject):
    '''This worker object will act to ensure that the QThreads are not within the Main Thread'''
    # def __init__(self, main_window, thread):
    def __init__(self, function, *args, **kwargs):
        '''takes in a function, and the arguments and keyword arguments for that function'''
        super(Worker, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.start.connect(self.run)

    start = QtCore.pyqtSignal(int)

    @QtCore.pyqtSlot()
    def run(self):
        self.function(*self.args, **self.kwargs)


def background(self):  # defines the background for each window
    """providing the background info for each window"""
    # Acquiring information about geometry
    self.PROJECT_DIR = os.path.dirname(os.path.abspath("__file__"))  # project directory
    self.CORE_DIR = os.path.join(self.PROJECT_DIR, 'core')
    self.SETTINGS_DIR = os.path.join(self.PROJECT_DIR, 'settings')
    if not os.path.exists(self.SETTINGS_DIR):
        os.mkdir(self.SETTINGS_DIR)
    self.IMG_DIR = os.path.join(self.PROJECT_DIR, 'img')
    self.setWindowIcon(QtGui.QIcon(os.path.join(self.IMG_DIR, 'cumc-crown.png')))  # declaring the icon image
    self.deskW, self.deskH = QtGui.QDesktopWidget().availableGeometry().getRect()[2:]  # gets the window resolution
    # self.setWindowState(QtCore.Qt.WindowMaximized) # will maximize the GUI
    self.setGeometry(0, 0, self.deskW*0.9, self.deskH*0.9)  # Sets the window size, 800x460 is the size of our window

    QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('GTK+'))


class MainWindow(QtGui.QWidget):  # defines the window class (main window)
    '''The function that will serve to create the main window of the GUI'''

    def __init__(self):  # initializes the main window
        super(MainWindow, self).__init__()
        # self.setGeometry(50, 50, 500, 300)

        background(self)  # acquires some features from the background function we defined earlier

        if getattr(sys, 'frozen', False):
            # frozen
            self.setWindowTitle(
                os.path.splitext(os.path.basename(sys.executable))[0] + " - Main Window")  # sets the title of the window

        else:
            # unfrozen
            self.setWindowTitle(
                os.path.splitext(os.path.basename(__file__))[
                    0] + " - Main Window")  # sets the title of the window

        self.settings_fname = os.path.join(self.SETTINGS_DIR, 'settings.json')

        self.home()  # runs the home function

    def home(self):  # defines the home function (the main window)
        '''a method that populates the QWidget with all the Widgets/layouts'''

        self.analyzing = False

        # ---------- directory layout --------------------------------
        file_layout = QtGui.QHBoxLayout()

        directory_label = QtGui.QLabel("Current H5 File:")

        self.filename = QtGui.QLineEdit()
        self.filename.setText("Choose a Set Directory!")
        self.filename.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        directory_with_label = QtGui.QHBoxLayout()
        directory_with_label.addWidget(directory_label)
        directory_with_label.addWidget(self.filename)

        self.choose_file = QtGui.QPushButton("Choose File")
        self.choose_file.clicked.connect(self.ChooseFile)

        file_layout.addWidget(self.choose_file)
        file_layout.addLayout(directory_with_label)

        # defining the plots as tabs
        tabs = QtGui.QTabWidget()

        # defining the peak frequency tab
        PeakFreq_tab = QtGui.QWidget()

        tabs.addTab(PeakFreq_tab, 'Peak Freqs')

        self.PeakFreqGraph = plt.figure(figsize=(30, 30))
        self.PeakFreqGraphCanvas = FigureCanvas(self.PeakFreqGraph)
        self.PeakFreqGraphAxis = self.PeakFreqGraph.add_axes([0.1, 0.05, 0.9, 0.9], frameon=False)

        PeakFreq_layout = QtGui.QVBoxLayout()
        PeakFreq_layout.addWidget(self.PeakFreqGraphCanvas)
        PeakFreq_tab.setLayout(PeakFreq_layout)

        # defining the T-F graph
        spectro_tab = QtGui.QWidget()

        tabs.addTab(spectro_tab, 'T-F')

        self.spectroGraph = plt.figure(figsize=(30, 30))
        self.spectroGraphCanvas = FigureCanvas(self.spectroGraph)
        self.spectroGraphAxis = self.spectroGraph.add_axes([0.1, 0.05, 0.9, 0.9], frameon=False)

        spectro_layout = QtGui.QVBoxLayout()
        spectro_layout.addWidget(self.spectroGraphCanvas)
        spectro_tab.setLayout(spectro_layout)

        # defining the band percentage graph
        band_perc_tab = QtGui.QWidget()

        tabs.addTab(band_perc_tab, 'Band Percentage')

        self.band_percGraph, self.band_percGraphAxis = plt.subplots(2, 4, figsize=(20, 15))
        # self.band_percGraph = plt.figure(figsize=(30, 30))
        self.band_percGraphCanvas = FigureCanvas(self.band_percGraph)
        # self.band_percGraphAxis = self.band_percGraph.add_axes([0.1, 0.2, 0.85, 0.75], frameon=False)

        band_perc_layout = QtGui.QVBoxLayout()
        band_perc_layout.addWidget(self.band_percGraphCanvas)
        band_perc_tab.setLayout(band_perc_layout)

        # defining the position bins graph
        position_bins_tab = QtGui.QWidget()

        tabs.addTab(position_bins_tab, 'Position Bins')

        self.position_binsGraph = plt.figure(figsize=(30, 30))
        self.position_binsGraphCanvas = FigureCanvas(self.position_binsGraph)
        self.position_binsGraphAxis = self.position_binsGraph.add_axes([0.05, 0.05, 0.9, 0.9], frameon=False)

        position_bins_layout = QtGui.QVBoxLayout()
        position_bins_layout.addWidget(self.position_binsGraphCanvas)
        position_bins_tab.setLayout(position_bins_layout)

        self.scrollbar = QtGui.QScrollBar(QtCore.Qt.Horizontal)

        # self.scrollbar.actionTriggered.connect(functools.partial(self.changeCurrentGraph, 'scroll'))

        graph_layout = QtGui.QVBoxLayout()
        graph_layout.addWidget(tabs)
        graph_layout.addWidget(self.scrollbar)

        # --------parameter layout-----------

        self.arena = QtGui.QComboBox()
        arena_options = ['BehaviorRoom', 'DarkRoom', 'Room4']

        for arena in arena_options:
            self.arena.addItem(arena)

        arena_label = QtGui.QLabel("Arena:")
        arena_layout = QtGui.QHBoxLayout()
        arena_layout.addWidget(arena_label)
        arena_layout.addWidget(self.arena)

        t_start_label = QtGui.QLabel('Start Time (s)')
        self.t_start = QtGui.QLineEdit()
        self.t_start.setAlignment(QtCore.Qt.AlignHCenter)
        t_start_layout = QtGui.QHBoxLayout()
        t_start_layout.addWidget(t_start_label)
        t_start_layout.addWidget(self.t_start)
        self.t_start.setText('N/A')

        t_stop_label = QtGui.QLabel('Stop Time (s)')
        self.t_stop = QtGui.QLineEdit()
        self.t_stop.setAlignment(QtCore.Qt.AlignHCenter)
        t_stop_layout = QtGui.QHBoxLayout()
        t_stop_layout.addWidget(t_stop_label)
        t_stop_layout.addWidget(self.t_stop)
        self.t_stop.setText('N/A')

        parameter_layout = QtGui.QHBoxLayout()
        parameter_layout.addStretch(1)
        for item in [t_start_layout, t_stop_layout, arena_layout]:
            if 'Layout' in item.__str__():
                parameter_layout.addLayout(item)
                parameter_layout.addStretch(1)
            else:
                parameter_layout.addWidget(item, 0, QtCore.Qt.AlignCenter)
                parameter_layout.addStretch(1)

        # ------ buttons -----------------------------
        self.run_btn = QtGui.QPushButton("Run", self)
        self.run_btn.setToolTip("Click to run the analysis (or press Ctrl+R)!")
        self.run_btn.setShortcut("Ctrl+R")
        self.run_btn.clicked.connect(self.analyze)  # connect the run button with the analyze() function

        quit_btn = QtGui.QPushButton("Quit", self)
        quit_btn.clicked.connect(self.close_app)
        quit_btn.setShortcut("Ctrl+Q")
        quit_btn.setToolTip('Click to quit (or press Ctrl+Q)')

        btn_layout = QtGui.QHBoxLayout()

        button_order = [self.run_btn, quit_btn]
        for button in button_order:
            btn_layout.addWidget(button)

        # ---------------- Version information ----------------------------------
        if getattr(sys, 'frozen', False):

            mod_date = time.ctime(os.path.getmtime(sys.executable))  # finds the modification date of the program

            vers_label = QtGui.QLabel(
                os.path.splitext(os.path.basename(sys.executable))[0] + " V1.0 - Last Updated: " + mod_date)

        else:

            mod_date = time.ctime(os.path.getmtime(__file__))  # finds the modification date of the program

            vers_label = QtGui.QLabel(os.path.splitext(os.path.basename(__file__))[0] + " V1.0 - Last Updated: " + mod_date)

        # ------------- layout ------------------------------

        layout = QtGui.QVBoxLayout()

        layout_order = [file_layout, graph_layout, parameter_layout, btn_layout]
        for order in layout_order:
            if 'Layout' in order.__str__():
                layout.addLayout(order)
                layout.addStretch(1)
            else:
                layout.addWidget(order, 0, QtCore.Qt.AlignCenter)
                layout.addStretch(1)

        layout.addStretch(1)  # adds stretch to put the version info at the buttom
        layout.addWidget(vers_label)  # adds the date modification/version number

        self.setLayout(layout)

        center(self)

        self.show()

    def initialize_parameters(self):

        self.t_start.setText('N/A')
        self.t_stop.setText('N/A')

        self.analyzing = False

        self.spectro_ds = None
        self.posy_interp = None
        self.posx_interp = None
        self.f_peak = None
        self.v = None

        self.PeakFreqGraphAxis.clear()
        self.position_binsGraphAxis.clear()

        for ax in self.band_percGraphAxis:
            ax.clear()

        self.spectroGraphAxis.clear()

        self.band_percGraphCanvas.draw()
        self.PeakFreqGraphCanvas.draw()
        self.position_binsGraphCanvas.clear()
        self.spectroGraphCanvas.draw()

    def raiseError(self, error_val):
        '''raises an error window given certain errors from an emitted signal'''
        if 'ChooseH5' in error_val:
            self.choice = QtGui.QMessageBox.question(self, "Error: Choose H5 File!",
                                                     "You have not chosen a directory yet to analyze!\n" +
                                                     "Please make sure to choose a directory before pressing 'Run'!\n",
                                                     QtGui.QMessageBox.Ok)

        elif 'H5ExistsError' in error_val:
            self.choice = QtGui.QMessageBox.question(self, "Error: H5 File Doesn't Exist!",
                                                     "You chose a file that doesn't exist!\n" +
                                                     "Please make sure to choose an existing .H5 file before pressing 'Run'!\n",
                                                     QtGui.QMessageBox.Ok)

    def analyze(self):

        if not self.analyzing:

            self.run_btn.setText('Stop')
            self.run_btn.setToolTip('Click to stop analysis.')  # defining the tool tip for the start button
            self.run_btn.clicked.disconnect()
            self.run_btn.clicked.connect(self.stop_analysis)

            current_file = self.filename.text()

            if 'Choose a Set' in current_file:
                self.choice = ''

                self.LogError.myGUI_signal.emit('ChooseH5')
                while self.choice == '':
                    time.sleep(0.1)
                return

            if not os.path.exists(current_file):
                self.choice = ''

                self.LogError.myGUI_signal.emit('H5ExistError')
                while self.choice == '':
                    time.sleep(0.1)
                return

            self.analyze_thread = QtCore.QThread()
            self.analyze_thread.start()
            self.analyze_thread_worker = Worker(Analyze, self)
            self.analyze_thread_worker.moveToThread(self.analyze_thread)
            self.analyze_thread_worker.start.emit(1)

    def stop_analysis(self):

        self.run_btn.setText('Run')
        self.analyze_thread.quit()

        self.run_btn.setToolTip('Click to start analysis.')  # defining the tool tip for the start button
        self.run_btn.clicked.disconnect()
        self.run_btn.clicked.connect(self.analyze)

    def close_app(self):
        '''Function that will close the app if the close button is pressed'''
        # pop up window that asks if you really want to exit the app ------------------------------------------------

        choice = QtGui.QMessageBox.question(self, "Quitting ",
                                            "Do you really want to exit?",
                                            QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if choice == QtGui.QMessageBox.Yes:
            sys.exit()  # tells the app to quit
        else:
            pass

    def ChooseFile(self):

        current_filename = str(QtGui.QFileDialog.getOpenFileName(
               self, 'Open File', '', 'H5 (*.H5)'))

        if current_filename != '':
            self.filename.setText(current_filename)
            self.initialize_parameters()


def Analyze(self):
    spatialSpectroAnalyze(self)
    self.stop_analysis()


class Communicate(QtCore.QObject):
    '''A custom pyqtsignal so that errors and popups can be called from the threads
    to the main window'''
    myGUI_signal = QtCore.pyqtSignal(str)


def center(self):
    """A function that centers the window on the screen"""
    frameGm = self.frameGeometry()
    screen = QtGui.QApplication.desktop().screenNumber(QtGui.QApplication.desktop().cursor().pos())
    centerPoint = QtGui.QApplication.desktop().screenGeometry(screen).center()
    frameGm.moveCenter(centerPoint)
    self.move(frameGm.topLeft())


def run():
    app = QtGui.QApplication(sys.argv)

    window = MainWindow()

    sys.exit(app.exec_())  # prevents the window from immediately exiting out


if __name__ == '__main__':
    run()  # the command that calls run()