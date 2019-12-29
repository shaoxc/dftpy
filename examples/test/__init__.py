# import sys
# from test.env import DFTPY_DATA_PATH

# DFTPY_DATA_PATH='./DATA/'
# sys.path.append(DFTPY_DATA_PATH)
# print(sys.path)
import sys
import os

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

