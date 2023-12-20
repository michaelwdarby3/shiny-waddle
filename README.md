## Installation

For this project, you'll need sklearn, itertools, pytorch, pytorch_geometric, torch_scatter, and torch_sparse installed, through pip or conda. The installations for each of those is a bit of a hassle to automate, so I'll refrain for so few requirements. The commands you should need are as such:

```
pip3 install torch
pip3 install torch_geometric
pip3 install torch_sparse
pip3 install torch_scatter
pip3 install sklearn
```

## How to Run

Once you have your packages installed, in your terminal of choice run `python3 main.py` and the program will run. This program was only tested on a machine with a GPU; I'm unclear if it will train slowly without one.

