import torch.distributed as dist
import torch

def main():
    torch.manual_seed(dist.get_rank())
    device = "cuda:{}".format(dist.get_rank())
    param = torch.FloatTensor(1,1).to(device)

    param.uniform_()
    print("Rank {}, before sync: {} at {}".format(dist.get_rank(),param.cpu().data.numpy(), param.device))
    dist.broadcast(param.data, 0)
    print("Rank {}, after sync : {} at {}".format(dist.get_rank(),param.cpu().data.numpy(), param.device))


if __name__ == '__main__':
    dist.init_process_group("mpi")
    main()
