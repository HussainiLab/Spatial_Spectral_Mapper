import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
#build_exe_options = {"packages": ["os"], "excludes": ["tkinter"]}

# GUI applications require a different base on Windows (the default is for a
# console application).
base = None
if sys.platform == "win32":
    base = "Win32GUI"

additional_imports = ['numpy.core._methods', 'numpy.lib.format', "matplotlib.backends.backend_tkagg",
                      'scipy.spatial']

packages = ['matplotlib', 'scipy', 'scipy.spatial']

setup(name="QuadrantGUI",
      version="1.0",
      description="QuadrantGUI breaks the position data into quadrants for analysis.",
      options={"build_exe": {'packages': packages, 'includes': additional_imports}},
      executables=[Executable("QuadrantGUI.py", base=base)])