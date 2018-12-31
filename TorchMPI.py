import os
import socket
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

dist.init_process_group(backend='mpi', world_size=4)
print('Hello from process {} (out of {})!'.format(dist.get_rank(), dist.get_world_size()))