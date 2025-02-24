import torch


class DisableTensorToDtype:
    def __enter__(self):
        self.original_to = torch.Tensor.to

        def modified_to(tensor, *args, **kwargs):
            # remove dtype from args if present
            args = [arg if not isinstance(arg, torch.dtype) else None for arg in args]
            if "dtype" in kwargs:
                kwargs.pop("dtype")
            return self.original_to(tensor, *args, **kwargs)

        torch.Tensor.to = modified_to

    def __exit__(self, *args, **kwargs):
        torch.Tensor.to = self.original_to
