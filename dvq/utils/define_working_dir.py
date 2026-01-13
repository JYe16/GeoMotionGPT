import os
import sys
project_root = '../../'
sys.path.append(project_root)

def define_working_dir():
    if os.path.isdir('/data/jackieye/'):
        return '/data/jackieye/llm-sensing-working-dir/'
    return '../working_dir'