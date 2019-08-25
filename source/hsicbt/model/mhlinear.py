from .. import *
from .block import *

class ModelLinear(nn.Module):

    def __init__(self, in_width=784, hidden_width=64, n_layers=5, atype='relu', 
        last_hidden_width=None, model_type='simple-dense', **kwargs):
        super(ModelLinear, self).__init__()
    
        block_list = []
        is_conv = False

        last_hw = hidden_width
        if last_hidden_width:
            last_hw = last_hidden_width
        
        for i in range(n_layers):
            _, block = get_primative_block(model_type, hidden_width, hidden_width, atype)
            block_list.append(block)

            self.input_layer    = makeblock_dense(in_width, hidden_width, atype)
            self.sequence_layer = nn.Sequential(*block_list)
            self.output_layer   = makeblock_dense(hidden_width, last_hw, atype)

        self.is_conv = is_conv
        self.in_width = in_width

    def forward(self, x):

        output_list = []
        x = x.view(-1, self.in_width)
        x = self.input_layer(x)
        output_list.append(x)
        
        for block in self.sequence_layer:
            x = block(x)
            output_list.append(x)
        x = self.output_layer(x)
        output_list.append(x)

        return x, output_list