import numpy as np
import torch, argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2tuple(s):
    return tuple(s.split('_'))

def s2ituple(s):
    return tuple(int(_s) for _s in s.split('_'))