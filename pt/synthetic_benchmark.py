from __future__ import print_function

import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import timeit, os
import numpy as np

parser = argparse.ArgumentParser(
              description = "PyTorch Synthetic Benchmarks",
              formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--fp16-allreduce", action="store_true", default=False,
                    help="use fp16 compression during allreduce")
parser.add_argument("--model", type=str, default="resnet50",
                    help="model to benchmark")
parser.add_argument("--batch-size", type=int, default=32,
                    help="input batch size")
parser.add_argument("--ip-addr", default="127.0.0.1", type=str,
                    help="launch node ip addr")
parser.add_argument("--num-warmup-batches", type=int, default=10,
                    help="number of warm-up batches that don't count towards benchmark")

parser.add_argument("--num-batches-per-iter", type=int, default=10,
                    help="number of batches per benchmark iteration")
parser.add_argument("--num-iters", type=int, default=10,
                    help="number of benchmark iterations")
parser.add_argument("--no-cuda", action="store_true", default=False,
                    help="disables CUDA training")
parser.add_argument("--backend", default="nccl", type=str,
                    help="nccl, mpi, horovod, or ddl")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if "MV2_COMM_WORLD_SIZE" in os.environ:
   world_size = int(os.environ["MV2_COMM_WORLD_SIZE"])
   world_rank = int(os.environ["MV2_COMM_WORLD_RANK"])
   local_rank = int(os.environ["MV2_COMM_WORLD_LOCAL_RANK"]) 
else:
   world_size = 1
   world_rank = 0
   local_rank = 0

if args.backend in ['horovod', 'ddl']:
      import horovod.torch as hvd
      hvd.init()
  else:
      import torch.distributed as dist
      if args.backend == 'nccl':
          import subprocess
          os.environ["MASTER_ADDR"] = args.ip_addr
          os.environ["MASTER_PORT"] = "23456"
          os.environ["WORLD_SIZE"] = str(world_size)
          os.environ["RANK"] = str(world_rank)
      dist.init_process_group(args.backend, rank=world_rank, world_size=world_size)
  if args.cuda:
      # Horovod: pin GPU to local rank
      torch.cuda.set_device(local_rank)

cudnn.benchmark = True

model = getattr(models, args.model)()
if args.cuda:
    model.cuda()

use_amp = True
optimizer = optim.SGD(model.parameters(), lr=0.01)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

if args.backend in ['horovod', 'ddl']:
    # Horovod: (optional) compression algorithm
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.non
    # Horovod: wrap optimizer with DistributedOptimizer
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters = model.named_parameters(),
                                         compression = compression)
    # Horovod: broadcast parameters and optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank = 0)
else:
    def average_gradients(model):
        # Gradient averaging
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size

data = torch.randn(args.batch_size, 3, 224, 224)
target = torch.LongTensor(args.batch_size).random_() % 1000
if args.cuda:
    data, target = data.cuda(), target.cuda()

def benchmark_step():
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        output = model(data)
        assert output.dtype is torch.float16
        loss = F.cross_entropy(output, target)
        assert loss.dtype is torch.float32
    scaler.scale(loss).backward()
    if args.backend in ["nccl", "mpi"]:
        average_gradients(model)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

def log(s, nl=True):
    if world_rank != 0:
        return
    print(s, end="\n" if nl else "")

log(f"Model: {args.model}")
log(f"Batch size: {args.batch_size}")
device = "GPU" if args.cuda else "CPU"
log(f"Number of {device}s: {world_size}")

# Warm-up
log("Running warmup...")
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log("Running benchmark...")
img_secs = []
for x in range(args.num_iters):
    pass
