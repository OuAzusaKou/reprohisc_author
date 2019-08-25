from .. import *

class ModelVanilla(nn.Module):

    def __init__(self, hidden_width=64, **kwargs):
        super(ModelVanilla, self).__init__()
        self.output = nn.Linear(hidden_width, 10)

    def forward(self, x):
        x = self.output(x)
        return F.log_softmax(x, dim=1)
