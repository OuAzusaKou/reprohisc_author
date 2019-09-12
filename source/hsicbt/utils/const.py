from .misc import get_current_timestamp
import os

TTYPE_STANDARD = 'backprop'
TTYPE_HSICTRAIN = 'hsictrain'
TTYPE_FORMAT = 'format'
TTYPE_UNFORMAT = 'unformat'
TTYPE_PLOT = 'plot'

FONTSIZE_TITLE  = 36
FONTSIZE_XLABEL = 34
FONTSIZE_YLABEL = 34
FONTSIZE_XTICKS = 22
FONTSIZE_YTICKS = 22
FONTSIZE_LEDEND = 28

DEBUG_MODE = 0
TIMESTAMP = 'HSICBT_TIMESTAMP'
if not os.environ.get(TIMESTAMP):
    os.environ[TIMESTAMP] = get_current_timestamp()
TIMESTAMP_CODE = os.environ[TIMESTAMP]
