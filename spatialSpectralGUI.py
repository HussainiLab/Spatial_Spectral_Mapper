from PyQt4 import QtGui, QtCore
import sys, os, time, datetime
from PIL import Image
from core.QuadrantFunctions import QuadrantAnalysis


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
    self.setGeometry(0, 0, self.deskW/3, self.deskH/2)  # Sets the window size, 800x460 is the size of our window

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

        self.LogError = Communicate()
        self.LogError.myGUI_signal.connect(self.raiseError)

        self.LogAppend = Communicate()
        self.LogAppend.myGUI_signal.connect(self.AppendLog)

        self.analyzing = False

        # ---------------logo --------------------------------

        cumc_logo = QtGui.QLabel(self)  # defining the logo image
        logo_fname = os.path.join(os.getcwd(), 'img', "QuadrantGUILogo.png")  # defining logo pathname
        im2 = Image.open(logo_fname)  # opening the logo with PIL
        # im2 = im2.resize((self.deskW,self.deskH), PIL.Image.ANTIALIAS)
        # im2 = im2.resize((100,100), Image.ANTIALIAS)
        logowidth, logoheight = im2.size  # acquiring the logo width/height
        logo_pix = QtGui.QPixmap(logo_fname)  # getting the pixmap
        cumc_logo.setPixmap(logo_pix)  # setting the pixmap
        cumc_logo.setGeometry(0, 0, logowidth, logoheight)  # setting the geometry

        # ---------- directory layout --------------------------------
        directory_layout = QtGui.QHBoxLayout()

        directory_label = QtGui.QLabel("Current Set Directory:")

        self.directory = QtGui.QLineEdit()
        self.directory.setText("Choose a Set Directory!")
        self.directory.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        directory_with_label = QtGui.QHBoxLayout()
        directory_with_label.addWidget(directory_label)
        directory_with_label.addWidget(self.directory)

        self.choose_directory = QtGui.QPushButton("Choose Directory")
        self.choose_directory.clicked.connect(self.ChooseDir)

        directory_layout.addWidget(self.choose_directory)
        directory_layout.addLayout(directory_with_label)

        # --------parameter layout-----------

        self.arena = QtGui.QComboBox()
        arena_options = ['BehaviorRoom', 'DarkRoom', 'Room4']

        for arena in arena_options:
            self.arena.addItem(arena)

        arena_label = QtGui.QLabel("Arena:")
        arena_layout = QtGui.QHBoxLayout()
        arena_layout.addWidget(arena_label)
        arena_layout.addWidget(self.arena)

        self.arena_shape = QtGui.QComboBox()
        arena_shape_options = ['Auto', 'Circle', 'Rectangular']

        for shape in arena_shape_options:
            self.arena_shape.addItem(shape)

        self.arena_shape.setCurrentIndex(self.arena_shape.findText("Circle"))

        arena_shape_label = QtGui.QLabel("Arena Shape:")
        arena_shape_layout = QtGui.QHBoxLayout()
        arena_shape_layout.addWidget(arena_shape_label)
        arena_shape_layout.addWidget(self.arena_shape)

        parameter_layout = QtGui.QHBoxLayout()
        parameter_layout.addStretch(1)
        parameter_layout.addLayout(arena_layout)
        parameter_layout.addStretch(1)
        parameter_layout.addLayout(arena_shape_layout)
        parameter_layout.addStretch(1)

        #------- log widget -------------------
        self.Log = QtGui.QTextEdit()
        log_label = QtGui.QLabel('Log: ')

        log_lay = QtGui.QHBoxLayout()
        log_lay.addWidget(log_label, 0, QtCore.Qt.AlignTop)
        log_lay.addWidget(self.Log)

        log_frame = QtGui.QFrame()
        log_frame.setFixedWidth(self.deskW / 2.1)
        log_frame.setLayout(log_lay)

        # ---------------------

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

        layout_order = [cumc_logo, directory_layout, parameter_layout, log_frame, btn_layout]
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

    def raiseError(self, error_val):
        '''raises an error window given certain errors from an emitted signal'''
        if 'ChooseDir' in error_val:
            self.choice = QtGui.QMessageBox.question(self, "Error: Choose Directory!",
                                                     "You have not chosen a directory yet to analyze!\n" +
                                                     "Please make sure to choose a directory before pressing 'Run'!\n",
                                                     QtGui.QMessageBox.Ok)

        elif 'DirExistsError' in error_val:
            self.choice = QtGui.QMessageBox.question(self, "Error: Directory Doesn't Exist!",
                                                     "You have a directory that doesn't exist!\n" +
                                                     "Please make sure to choose an existing directory before pressing 'Run'!\n",
                                                     QtGui.QMessageBox.Ok)

    def AppendLog(self, message):
        '''A function that will append the Log field of the main window (mainly
        used as a slot for a custom pyqt signal)'''
        self.Log.append(message)

    def analyze(self):

        if not self.analyzing:

            self.run_btn.setText('Stop')
            self.run_btn.setToolTip('Click to stop analysis.')  # defining the tool tip for the start button
            self.run_btn.clicked.disconnect()
            self.run_btn.clicked.connect(self.stop_analysis)

            current_directory = self.directory.text()

            if 'Choose a Set' in current_directory:
                self.choice = ''
                self.LogAppend.myGUI_signal.emit(
                    '[%s %s]: Error: Choose a directory!' % (str(datetime.datetime.now().date()),
                                                                         str(datetime.datetime.now().time())[:8]))

                self.LogError.myGUI_signal.emit('ChooseDir')
                while self.choice == '':
                    time.sleep(0.1)
                return

            if not os.path.exists(current_directory):
                self.choice = ''
                self.LogAppend.myGUI_signal.emit(
                    '[%s %s]: Error: Directory doesn\'t exist!' % (str(datetime.datetime.now().date()),
                                                             str(datetime.datetime.now().time())[:8]))
                self.LogError.myGUI_signal.emit('DirExistError')
                while self.choice == '':
                    time.sleep(0.1)
                return

            arena = self.arena.currentText()

            arena_shape = self.arena_shape.currentText()
            if arena_shape == 'Auto':
                arena_shape = None

            self.analyze_thread = QtCore.QThread()
            self.analyze_thread.start()
            self.analyze_thread_worker = Worker(Analyze, self, current_directory, arena, arena_shape)
            self.analyze_thread_worker.moveToThread(self.analyze_thread)
            self.analyze_thread_worker.start.emit(1)

    def stop_analysis(self):

        self.run_btn.setText('Run')
        self.analyze_thread.quit()
        self.analyzing = False
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

    def ChooseDir(self):

        current_directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))

        if current_directory != '':
            self.directory.setText(current_directory)


def Analyze(self, set_directory, arena, arena_shape):

    self.analyzing = True
    QuadrantAnalysis(self, set_directory, arena, arena_shape=arena_shape)
    self.analyzing = False
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