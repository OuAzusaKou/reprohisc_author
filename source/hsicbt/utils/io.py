from .. import *
from .color import *
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