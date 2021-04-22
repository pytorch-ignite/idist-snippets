import argparse
import time
import ignite.distributed as idist
from ignite.engine import Engine


# This method will be called for each of the processes
def _mp_train(local_rank):
    # A training step printing process information and underlying data
    def _train_step(e, batch):
        print(
            f"Process {idist.get_rank()}/{idist.get_world_size()} : Epoch {e.state.epoch} - {e.state.iteration} : batch={batch}")
        # This is a synchronization point where we are waiting all the process to finish the previous commands
        idist.barrier()
        if idist.get_rank() == 0:
            time.sleep(2)

    # Define dummy input data for sake of simplicity
    batch_data = [0, 1, 2]
    # Running the _train_step function on whole batch_data iterable only once
    trainer = Engine(_train_step)
    trainer.run(batch_data, max_epochs=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training-Step")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--nproc_per_node", type=int)
    args = parser.parse_args()

    # idist from ignite handles multiple backend (gloo, nccl, horovod, xla)
    # and launcher (torch.distributed.launch, horovodrun, slurm)
    kwargs = dict()
    if args.nproc_per_node is not None:
        kwargs["nproc_per_node"] = args.nproc_per_node
    with idist.Parallel(backend=args.backend, **kwargs) as parallel:
        parallel.run(_mp_train)
