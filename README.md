# Spatial_Spectral_Mapper
## Map frequency power spectrums from murine neural eeg/egf data as a function of position

### Installation
**Notice**: 
Spatial_Spectral_Mapper was written using the PyQt5 framework in python. Python 3.8 is reccomended. You can install v3.8 of python at www.python.org/downloads

----

#### Clone the repository
You can clone this repository into your local machine using ```git clone``` or alternatively download the zip file directly from the web.

----

#### Using Anaconda
If Anaconda is your primary python management software, you can run Spatial Spectral Mapper using the following steps:
1. Create a new virtual environment for python 3.8 using Anacondas GUI-based environment manager (reccomended)
2. Do ***NOT*** use pip install for requirements.txt. Pip has been known to break dependencies in Anaconda. Instead, use the built in package index finder to grab the packages in requirements.txt and download them manually. *Note, anaconda will probably not list PyQt5 as an available package. Instead, you can download the Qt compatability packages it lists.)*
3. At this point, you may either use an IDE of choice such as Spyder to run main.py (in the src directory), or you can use Anaconda's command line to run ```python -m main.py```

----

#### Using Visual Studio
If Visual Studio is your preffered IDE, you can run this program doing the following:
1. Click the 'python environments' button.

2. In the Python environments window, click 'Add Environment'
3. Name your virtual environment, and choose Python 3.8 as your interpreter.
4. Make sure that the 'Install packages from file' field points to the requirements.txt file path
5. Create your new virtual environment. You can now run main.py 
	(located in src) once you have selected your newly created environment. 

----

#### Using the command line and virtualenv on Windows
If the command line is your primary method of python environment management, you can run this program using the following steps:
1. Make sure that python version 3.8 is installed on your system. On windows, you can check all the current versions of python installed using ```python -0``` through windows powershell.
2. Navigate to the PRISM parent folder. If you haven't yet installed virtualenv, you can do so using ```python -3.8 -m pip install virtualenv```
3. You can now create a new virtualenv using ```python -3.8 -m virtualenv my_new_cool_env```
4. Activate your virtualenv by typing ```my_new_cool_env/Scripts/activate```. Note that you should be in the directory where *my_new_cool_env* is located.
5. Once this environment has been activated, you can safely install pip dependencies without worrying about affecting the base environment. Use ```pip install -r requirements.txt``` after navigating to the directory where it is stored.
6. You can now run Spatial Spectral Mapper using ```python main.py```

----

#### Excel requirement
Spatial Spectral Mapper relies on the xlwings package to populate data into excel sheets. As such, a working installation of excel is required for this program to run. 

