import sys
import os
print("Python Executable:", sys.executable)
print("Sys Path:", sys.path)
try:
    import gm
    print("GM Module Found:", gm.__file__)
except ImportError as e:
    print("GM Import Failed:", e)

try:
    import pandas
    print("Pandas Version:", pandas.__version__)
except ImportError:
    print("Pandas Import Failed")
