from . import *


def task_assigner(ttype):
    func = None
    if ttype == TTYPE_HSICTRAIN:
        func = training_hsic
    elif ttype == TTYPE_STANDARD:
        func = training_standard
    elif ttype == TTYPE_FORMAT:
        func = training_format
    return func

