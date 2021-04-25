import os
import socket
import argparse
import ignite.distributed as idist



def main_fn(_):
    hostname = socket.gethostname()

    for current in range(idist.get_world_size()):
        if idist.get_rank() == current:
            addr = f"http://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
            print(f"[{addr}] hello from [{hostname}:{idist.backend()}] "
                  f"process {idist.get_rank()}/{idist.get_world_size()}")
        idist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("single-node")
    parser.add_argument("--backend", type=str, default="nccl")
    args = parser.parse_args()

    # idist from ignite handles multiple backend (gloo, nccl, horovod, xla)
    # and launcher (slurm, torch.distributed.launch)
    with idist.Parallel(backend=args.backend) as parallel:
        parallel.run(main_fn)
