import sys
import os

# اضافه کردن مسیر پروژه به sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train
from test import test

if __name__ == "__main__":
    train()
    test()