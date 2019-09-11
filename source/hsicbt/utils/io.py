from .. import *
from .color import *
from .misc import get_current_timestamp
import yaml

def load_yaml(filepath):

    with open(filepath, 'r') as stream:
        try:
            data = yaml.load(stream, yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    print_highlight("Loaded [{}]".format(filepath))
    return data
    
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print_highlight("Saved [{}]".format(filepath))

def load_model(filepath):
    model = torch.load(filepath)
    print_highlight("Loaded [{}]".format(filepath))
    return model

def save_logs(logs, filepath):
    timestamp = get_current_timestamp()
    filename = os.path.basename(filepath)
    dirname = os.path.dirname(filepath)
    filename_time = "{}_{}.npy".format(timestamp, os.path.splitext(filename)[0])
    timestamp_path = os.path.join(dirname, filename_time)
    np.save(timestamp_path, logs)
    
    if os.path.exists(filepath):
        os.remove(filepath)
        
    os.symlink(timestamp_path, filepath)
    print_highlight("Saved [{}]".format(timestamp_path), ctype="blue")
    print_highlight("Symlink [{}]".format(filepath), ctype="blue")

def load_logs(filepath):
    logs = np.load(filepath, allow_pickle=True)[()]
    print_highlight("Loaded [{}]".format(filepath), ctype="blue")
    return logs
