import torch
import random

# print(random.randint(0,1,(30,)))
x = torch.randint(0,1,(3,))
x = torch.cat([torch.tensor([100]),
           torch.tensor([200]),
           x], dim=0)
print(x.shape)