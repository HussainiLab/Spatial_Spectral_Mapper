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
        self.running = True

    start = QtCore.pyqtSignal(int)

    @QtCore.pyqtSlot()
    def run(self):
        self.function(*self.args, **self.kwargs)
        self.running = False


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

        self.LogError = Communicate()
        self.LogError.myGUI_signal.connect(self.raiseError)

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

        self.band_percGraph, self.band_percGraphAxis = plt.subplots(2, 5, figsize=(20, 15))
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
        self.scrollbar.setPageStep(60)
        self.scrollbar.setSingleStep(60)
        self.scrollbar.valueChanged.connect(self.scrollChange)
        # self.scrollbar.sliderReleased.connect(self.scrollChange)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(0)

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

        geometry_label = QtGui.QLabel('Geometry (rows,cols)')
        self.geometry_rows = QtGui.QLineEdit()
        self.geometry_rows.setAlignment(QtCore.Qt.AlignHCenter)
        self.geometry_rows.setText('32')
        self.geometry_rows.textChanged.connect(self.get_geometry)

        self.geometry_cols = QtGui.QLineEdit()
        self.geometry_cols.setAlignment(QtCore.Qt.AlignHCenter)
        self.geometry_cols.setText('32')
        self.geometry_cols.textChanged.connect(self.get_geometry)

        geometry_values_layout = QtGui.QHBoxLayout()
        geometry_values_layout.addWidget(self.geometry_rows)
        geometry_values_layout.addWidget(self.geometry_cols)

        geometry_layout = QtGui.QHBoxLayout()
        geometry_layout.addWidget(geometry_label)
        geometry_layout.addLayout(geometry_values_layout)

        t_start_label = QtGui.QLabel('Start Time (s)')
        self.t_start = QtGui.QLineEdit()
        self.t_start.setAlignment(QtCore.Qt.AlignHCenter)
        t_start_layout = QtGui.QHBoxLayout()
        t_start_layout.addWidget(t_start_label)
        t_start_layout.addWidget(self.t_start)
        self.t_start.setText('N/A')
        self.t_start.textChanged.connect(functools.partial(self.changeSliceSize, 'times'))

        t_stop_label = QtGui.QLabel('Stop Time (s)')
        self.t_stop = QtGui.QLineEdit()
        self.t_stop.setAlignment(QtCore.Qt.AlignHCenter)
        t_stop_layout = QtGui.QHBoxLayout()
        t_stop_layout.addWidget(t_stop_label)
        t_stop_layout.addWidget(self.t_stop)
        self.t_stop.setText('N/A')
        self.t_stop.textChanged.connect(functools.partial(self.changeSliceSize, 'times'))

        slice_size_label = QtGui.QLabel('Slice Size (s)')
        self.slice_size = QtGui.QLineEdit()
        self.slice_size.setAlignment(QtCore.Qt.AlignHCenter)
        slice_size_layout = QtGui.QHBoxLayout()
        slice_size_layout.addWidget(slice_size_label)
        slice_size_layout.addWidget(self.slice_size)
        self.slice_size.setText('N/A')
        self.slice_size.textChanged.connect(functools.partial(self.changeSliceSize, 'slice'))

        scroll_step_label = QtGui.QLabel('Scroll Step (s)')
        self.scroll_step = QtGui.QLineEdit()
        self.scroll_step.setAlignment(QtCore.Qt.AlignHCenter)
        scroll_step_layout = QtGui.QHBoxLayout()
        scroll_step_layout.addWidget(scroll_step_label)
        scroll_step_layout.addWidget(self.scroll_step)
        self.scroll_step.setText('60')
        self.scroll_step.textChanged.connect(self.changeScrollStep)

        parameter_layout = QtGui.QHBoxLayout()
        parameter_layout.addStretch(1)
        for item in [t_start_layout, t_stop_layout, slice_size_layout, scroll_step_layout, geometry_layout, arena_layout]:
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
        self.run_btn.clicked.connect(self.run)  # connect the run button with the analyze() function

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

        self.initialize_parameters()

        self.show()

    def get_geometry(self):

        try:
            rows = int(self.geometry_rows.text())
            cols = int(self.geometry_cols.text())

            if rows >= 1 and cols >= 1:
                # need positive integers (no zeros)
                self.geometry = (rows, cols)
            else:
                self.geometry = False
        except:
            self.geometry = False

        return self.geometry

    def changeScrollStep(self):

        try:
            step_value = int(self.scroll_step.text())
            self.scrollbar.setPageStep(step_value)
            self.scrollbar.setSingleStep(step_value)
            # self.scrollbar.setMinimum(0)
            # self.scrollbar.setMaximum(self.max_t - self.scrollbar_width)
        except ValueError:
            print(self.scroll_step.text())

    def scrollChange(self):

        if self.data_loaded:
            current_start = self.scrollbar.value()
            current_stop = self.scrollbar.value() + self.scrollbar_width

            self.t_start.setText(str(current_start))
            self.t_stop.setText(str(current_stop))

    def changeGraphs(self):
        """Continuously checks if the graphs need to be updated"""
        while not self.data_loaded:
            if not self.analyzing:
                error = self.analyze()
                if 'Abort' in error:
                    break
            time.sleep(0.1)

        while self.data_loaded:

            if self.analyzing:
                time.sleep(0.1)
                continue

            # if self.analyze_thread.isRunning():
            if self.analyze_thread_worker.running:
                time.sleep(0.1)
                continue

            try:
                current_start = float(self.t_start.text())
                current_stop = float(self.t_stop.text())
            except ValueError:
                time.sleep(0.1)
                continue

            if np.float32(current_start) == np.float32(self.previous_t_start) and np.float32(current_stop) == np.float32(self.previous_t_stop):
                """This is the current graph, skip"""
                time.sleep(0.1)
                continue

            #if current_stop - current_start != float(self.slice_size.text()):
            #    """The values don't match up yet, probably still updating, skip"""
            #    continue

            error = self.analyze()  # re-create the graphs
            if 'Abort' in error:
                break

    def changeSliceSize(self, source):
        """If the slice field is changed, it this method will change the t-stop value based off of the slice width,
        if the start or stop time was changed, it will change the slice field to match"""

        if self.data_loaded:
            if 'slice' in source:
                '''
                self.scrollbar_width = float(self.slice_size.text())
                t_stop_value = float(self.t_start.text()) + self.scrollbar_width
                if t_stop_value > self.max_t:
                    t_stop_value = self.max_t
                    self.scrollbar_width = self.max_t
                    self.slice_size.setText(str(t_stop_value))
                    self.t_start.setText(str(self.max_t - self.scrollbar_width))

                self.t_stop.setText(str(t_stop_value))
                '''
                self.scrollbar_width = float(self.slice_size.text())
                self.scrollbar.setMinimum(0)
                self.scrollbar.setMaximum(self.max_t - self.scrollbar_width)
            elif 'times' in source:
                """This will change the size size to match"""
                try:

                    new_size = float(self.t_stop.text()) - float(self.t_start.text())
                    t_stop_value = float(self.t_start.text()) + new_size
                    if t_stop_value > self.max_t:
                        t_stop_value = self.max_t
                        self.t_stop.setText(str(t_stop_value))
                    else:
                        self.slice_size.setText(str(new_size))
                        self.scrollbar_width = new_size
                        self.scrollbar.setMinimum(0)
                        self.scrollbar.setMaximum(self.max_t - self.scrollbar_width)
                except ValueError:
                    # invalid time number most likely
                    pass

    '''
    def SliceGraph(self, source):
        if not self.data_loaded:
            return

        if 'text' in source:

            if self.t_start.text() < 0:
                self.t_start.setText('0')

            if self.t_stop.text() > self.max_t:
                self.t_stop.setText(str(self.max_t))

            self.scrollbar_width = float(self.t_start.text()) - float(self.t_stop.text())
            self.slice_size.setText = str(self.scrollbar_width)

            self.scrollbar.setMinimum(0)
            self.scrollbar.setMaximum(self.max_t - self.scrollbar_width)
    '''

    def initialize_parameters(self):

        if self.run_btn.text() != 'Run':
            self.stop_analysis()

        self.labels = None
        self.spectro_ydotted = None
        self.spectro_yticks = None

        # self.geometry_rows.setText('32')
        # self.geometry_cols.setText('32')
        if not hasattr(self, 'bandpercGraphColorbar'):
            self.bandpercGraphColorbar = []
        if not hasattr(self, 'PeakFreqGraphColorbar'):
            self.PeakFreqGraphColorbar = None
        if not hasattr(self, 'spectroGraphColorbar'):
            self.spectroGraphColorbar = None

        self.current_start = None
        self.current_stop = None
        self.previous_t_start = -1
        self.previous_t_stop = -2
        self.previous_arena = None
        self.previous_geometry = None

        self.spatial_spectral_freq_bounds = {}
        self.t_axis_bool = None
        self.zero_t_start = None
        self.appended_zeros = True
        self.eeg = None
        self.eeg_t = None
        self.geometry = False
        self.t_start.setText('N/A')
        self.t_stop.setText('N/A')
        self.max_t = None
        self.previousValue = 0
        self.data_loaded = False
        self.scrollbar_width = None
        self.analyzing = False
        self.spectro_ds = None
        self.posy_interp = None
        self.posx_interp = None
        self.f_peak = None
        self.v = None
        self.frequency_boundaries = None
        self.band_order = None
        self.spectro_extent_vals = None
        self.extent_vals = None

        self.PeakFreqGraphAxis.clear()
        self.position_binsGraphAxis.clear()

        for ax in self.band_percGraphAxis.flatten():
            ax.clear()

        self.spectroGraphAxis.clear()

        self.band_percGraphCanvas.draw()
        self.PeakFreqGraphCanvas.draw()
        self.spectroGraphCanvas.draw()
        self.position_binsGraphCanvas.draw()

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

        elif 'GeometryError' in error_val:
            self.choice = QtGui.QMessageBox.question(self, "Error: Geometry!",
                                                     "Make sure you are using a positive integer for both geometry values!\n",
                                                     QtGui.QMessageBox.Ok)

    def changed_parameters(self):
        """this checks if the parameters have changed since the last run"""
        # current_start = self.current_start
        # current_stop = self.current_stop
        try:
            current_start = float(self.t_start.text())
            current_stop = float(self.t_stop.text())
        except ValueError:
            return False

        current_geometry = self.get_geometry()
        current_arena = self.arena.currentText()

        if current_start != self.previous_t_start or current_stop != self.previous_t_stop or current_geometry != \
                self.previous_geometry or current_arena != self.previous_arena:
            return True
        return False

    def run(self):
        # self.analyze()  # run the first analysis

        self.run_thread = QtCore.QThread()
        self.run_thread.start()
        self.run_thread_worker = Worker(self.changeGraphs)
        self.run_thread_worker.moveToThread(self.run_thread)
        self.run_thread_worker.start.emit(1)

    def analyze(self):

        if not self.analyzing:

            current_file = self.filename.text()

            # check if a filename was chosen
            if 'Choose a Set' in current_file:
                self.choice = ''

                self.LogError.myGUI_signal.emit('ChooseH5')
                while self.choice == '':
                    time.sleep(0.1)
                self.stop_analysis()
                # time.sleep(0.1)
                return 'Abort'

            # check if the filename exists
            if not os.path.exists(current_file):
                self.choice = ''

                self.LogError.myGUI_signal.emit('H5ExistError')
                while self.choice == '':
                    time.sleep(0.1)
                self.stop_analysis()
                return 'Abort'

            self.geometry = self.get_geometry()
            # check if the geometry is valid
            if self.geometry is False:

                self.choice = ''

                self.LogError.myGUI_signal.emit('GeometryError')
                while self.choice == '':
                    time.sleep(0.1)
                self.stop_analysis()
                return 'Abort'

            # check if any parameters have changed
            if self.data_loaded:
                if not self.changed_parameters():
                    return 'Abort'

            self.run_btn.setText('Stop')
            self.run_btn.setToolTip('Click to stop analysis.')  # defining the tool tip for the start button
            self.run_btn.clicked.disconnect()
            self.run_btn.clicked.connect(self.stop_analysis)

            if not self.data_loaded:
                self.analyze_thread = QtCore.QThread()
                self.analyze_thread.start()
                self.analyze_thread_worker = Worker(Analyze, self, current_start=0, current_stop=None)
            else:

                current_start = float(self.t_start.text())
                current_stop = float(self.t_stop.text())

                if np.float32(current_start) == np.float32(self.previous_t_start) and np.float32(
                        current_stop) == np.float32(self.previous_t_stop):
                    return ''
                self.analyze_thread = QtCore.QThread()
                self.analyze_thread.start()

                self.analyze_thread_worker = Worker(Analyze, self, current_start=float(self.t_start.text()),
                                                    current_stop=float(self.t_stop.text()))
            self.analyze_thread_worker.moveToThread(self.analyze_thread)
            self.analyze_thread_worker.start.emit(1)
            return ''

    def stop_analysis(self):

        self.run_btn.setText('Run')
        try:
            self.run_btn.clicked.disconnect()
            self.run_btn.clicked.connect(self.run)
        except:
            pass

        # self.run_thread.deleteLater()
        try:
            # self.run_thread.quit()
            self.run_thread.terminate()
            # self.analyze_thread.quit()
            self.analyze_thread.terminate()
            '''
            self.LogAppend.myGUI_signal.emit(
                '[%s %s]: Conversion terminated!' %
                (str(datetime.datetime.now().date()),
                 str(datetime.datetime.now().time())[:8]))'''
        except AttributeError:
            pass

        # self.analyzing = False
        self.run_btn.setToolTip('Click to start analysis.')  # defining the tool tip for the start button

    def close_app(self):
        '''Function that will close the app if the close button is pressed'''
        # pop up window that asks if you really want to exit the app ------------------------------------------------

        choice = QtGui.QMessageBox.question(self, "Quitting ",
                                            "Do you really want to exit?",
                                            QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if choice == QtGui.QMessageBox.Yes:

            self.stop_analysis()
            time.sleep(0.1)
            sys.exit()  # tells the app to quit
        else:
            pass

    def ChooseFile(self):

        current_filename = str(QtGui.QFileDialog.getOpenFileName(
               self, 'Open File', '', 'H5 (*.H5)'))

        if current_filename != '':
            self.filename.setText(current_filename)
            self.initialize_parameters()


def Analyze(self, current_start, current_stop=None):
    spatialSpectroAnalyze(self, current_start=current_start, current_stop=current_stop)
    # self.stop_analysis()


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