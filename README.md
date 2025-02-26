# kdsml
Knowledge Distillation Framework to enhance performance of Small Language Model (SLM)

> To run on distributed environemnt
```shell
torchrun --nproc_per_node=1 ddp_main.py
```
> To run in backgroun using `nohup`

```shell
nohup torchrun --nproc_per_node=4 ddp_main.py > logs/output.log 2>&1 &
```