
import runpy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

runpy.run_module('m2se_vtts.run', run_name='__main__', alter_sys=True)
