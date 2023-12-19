## Installation

For this project, you'll need sklearn, pytorch, pytorch_geometric, torch_scatter, and torch_sparse installed, through pip or conda. The installations for each of those is a bit of a hassle to automate, so I'll refrain for so few requirements. The commands you should need are as such:

```
pip3 install torch
pip3 install torch_geometric
pip3 install torch_sparse
pip3 install torch_scatter
pip3 install sklearn
```

## How to Run

Once you have your packages installed, in your terminal of choice run `python3 main.py` and the program will run. This program was only tested on a machine with a GPU; I'm unclear if it will train slowly without one.


## Caveats

The torch_geometric package is a bit finnicky, and so much of the time the program will fail due to inconsistent behavior from torch_geometric. I've noted as high as a 60% fail rate depending on my selected hyperparameters; this bug is identifiable by the sizes of tensors not matching up, and you should be able to just run it again and you'll get a successful run soon enough.

The DataLoaders and DataSamplers proved to be a hassle to get to work right with the nested cross-validation procedure, in part due to my inexperience with both classes.

