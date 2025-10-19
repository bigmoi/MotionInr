from torch import  nn


#a empty tokenizer that returns the input as it is ,to be used when no tokenizer is needed
class empty_tokenizer(nn.modules):
    def __init__(self,*args, **kwargs):
        super().__init__()

    def forward(self, x, *args,**kwargs):
        return  x

