How to run these code snippets:
===

The code snippets highlight the API's specificities of each of the distributed backends on the same use case as compared to the `idist` API. Torch native code is available for DDP, Horovod, and for XLA/TPU devices. 

One can run each of the code snippets independently. 

### With `torch.multiprocessing.spawn` 
In this case `idist Parallel` is using the native torch `torch.multiprocessing.spawn` method under the hood in order to run the distributed configuration. Here `nproc_per_node` is passed as a spawn argument.

- Running multiple distributed configurations with one code. Source: [ignite_idist.py](https://github.com/pytorch-ignite/idist-snippets/blob/master/ignite_idist.py):
 ```commandline
# Running with gloo
python -u ignite_idist.py --nproc_per_node 2 --backend gloo

# Running with nccl
python -u ignite_idist.py --nproc_per_node 2 --backend nccl

# Running with horovod with gloo controller ( gloo or nccl support )
python -u ignite_idist.py --backend horovod --nproc_per_node 2

# Running on xla/tpu
python -u ignite_idist.py --backend xla-tpu --nproc_per_node 8 --batch_size 32
```

### With Distributed launchers
PyTorch-Ignite's `idist Parallel`  context manager is also compatible with multiple distributed launchers.



#### With torch.distributed.launch

Here we are using the `torch.distributed.launch` script in order to spawn the processes:

```commandline
python -m torch.distributed.launch --nproc_per_node 2 --use_env ignite_idist.py --backend gloo
```

#### With horovodrun

```commandline
horovodrun -np 16 -H hostname1:8,hostname2:8 python ignite_idist.py --backend horovod
```

 
In order to run  this example and to avoid the installation procedure, you can pull one of PyTorch-Ignite's [docker image with pre-installed Horovod](https://github.com/pytorch/ignite/blob/master/docker/hvd/Dockerfile.hvd-base). It will include Horovod with `gloo` controller and `nccl` support.

```commandline
docker run --gpus all -it -v $PWD:/workspace/project --network=host --shm-size 16G pytorchignite/hvd-vision:latest /bin/bash
cd project
...
```


#### With slurm

The same result can be achieved  by using `slurm` without any modification to the code:

```commandline
srun --nodes=2
     --ntasks-per-node=2 
     --job-name=pytorch-ignite 
     --time=00:01:00  
     --partition=gpgpu 
     --gres=gpu:2
     --mem=10G 
     python ignite_idist.py --backend nccl
```

or using `sbatch script.bash` with the script file `script.bash`:
```bash
#!/bin/bash
#SBATCH --job-name=pytorch-ignite
#SBATCH --output=slurm_%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:01:00
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:2
#SBATCH --mem=10G

srun python ignite_idist.py --backend nccl
```

### Running Torch native run methods
In order to run the same training loop on different backends without `idist` you would have to use the different native torch snippets and associate a specific launch method for each of them. 

#### Torch native DDP
- Run the `torch native` snippet with different backends:
```commandline
# Running with gloo 
python -u torch_native.py --nproc_per_node 2 --backend gloo

# Running with nccl
python -u torch_native.py --nproc_per_node 2 --backend nccl
```

#### Horovod

- Run `horovod native` with `gloo` controller and `nccl`/`gloo` supports

```commandline
# Running with horovod with gloo controller ( gloo or nccl support )
python -u torch_horovod.py --nproc_per_node 2t

```

#### XLA/TPU devices

- Run `torch xla native` snippet on tpa/xlu with :
```commandline
# Run with a default of 8 processes 
python -u torch_xla_native.py
```
