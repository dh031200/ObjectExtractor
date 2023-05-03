__version__ = '4.0.1'

import sys
import os

sys.path.append(f'{os.getcwd()}/ultralytics')

import argparse
from utils.yolo import Model
from utils.data import Data


def processing(current, total):
    pattern = '⠋⠓⠚⠙'
    dots = ['.  ', '.. ', '...']
    print(f' {pattern[current % 4]} Processing{dots[current // 10 % 3]}  {current}/{total}', end='\r')


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--source', default='https://youtu.be/Zgi9g1ksQHc', help='source')
    parser.add_argument('--model', default='yolov8n.pt', help='model')
    parser.add_argument('--show', action='store_true', help='display results')
    parser.add_argument('--save-vid', action='store_true', help='save results video')
    parser.add_argument('--verbose', action='store_true', help='print verbose')

    args = parser.parse_args()
    return args


__all__ = '__version__', 'Model', 'Data', 'parse_args', 'processing'
