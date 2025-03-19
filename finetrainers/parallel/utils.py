import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor


def dist_reduce(x: torch.Tensor, reduceOp: str, mesh: torch.distributed.device_mesh.DeviceMesh) -> float:
    if isinstance(x, torch.distributed.tensor.DTensor):
        # functional collectives do not support DTensor inputs
        x = x.full_tensor()
    assert x.numel() == 1  # required by `.item()`
    return funcol.all_reduce(x, reduceOp=reduceOp, group=mesh).item()


def dist_max(x: torch.Tensor, mesh: torch.distributed.device_mesh.DeviceMesh) -> float:
    return dist_reduce(x, reduceOp=torch.distributed.distributed_c10d.ReduceOp.MAX.name, mesh=mesh)


def dist_mean(x: torch.Tensor, mesh: torch.distributed.device_mesh.DeviceMesh) -> float:
    return dist_reduce(x, reduceOp=torch.distributed.distributed_c10d.ReduceOp.AVG.name, mesh=mesh)
