from .. import *

from ..task.task_general     import *
from ..task.task_hsicsolve   import *
from ..task.task_niddle      import *
from ..task.task_variedact   import *
from ..task.task_variedep    import *
from ..task.task_varieddepth import *
from ..task.task_sigmacomb   import *
from ..task.task_varieddim   import *

def job_execution(config_dict):

    if config_dict['task'] == 'standard-train':
        out_batch, out_epoch = training_standard(config_dict)

    elif config_dict['task'] == 'hsic-train':
        out_batch, out_epoch = training_hsic(config_dict)

    elif config_dict['task'] == 'format-train':
        out_batch, out_epoch = training_format(config_dict)

    elif config_dict['task'] == 'niddle':
        task_niddle_func(config_dict)

    elif config_dict['task'] == 'general':
        task_general_func(config_dict)

    elif config_dict['task'] == 'hsic-solve':
        task_hsicsolve_func(config_dict)
        
    elif config_dict['task'] == 'varied-activation':
        task_variedact_func(config_dict)

    elif config_dict['task'] == 'sigma-combined':
        task_sigmacomb_func(config_dict)
        
    elif config_dict['task'] == 'varied-depth':
        task_varieddepth_func(config_dict)

    elif config_dict['task'] == 'varied-epoch':
        task_variedep_func(config_dict)

    elif config_dict['task'] == 'varied-dim':
        task_varieddim_func(config_dict)

    else:
        raise ValueError("Unknown given task [{}], please check \
            hsicbt.dispatcher.job_execution".format(config_dict['task']))
    
