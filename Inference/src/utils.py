import sys
import os


def disbalePrint():
    sys.stdout = open(os.devnull, "w")


def enablePrint():
    sys.stdout = sys.__stdout__
