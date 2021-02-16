from __future__ import annotations
import torch

def select_device(id: int) -> torch.device:
    """
    Return torch.device reference for the device CPU or GPU chosen to carry out the analyses.
    """
    force_cpu = False
    if id == -1:
        force_cpu = True

    # Detect cuda
    cuda: bool = False if force_cpu else torch.cuda.is_available()

    # Torch Device Class Object
    device: torch.device = torch.device("cuda:{}".format(id) if cuda else "cpu")

    if not cuda:
        print("Using CPU")
    if cuda:
        # bytes to MB unit conversion.
        conversion_factor = 1024 ** 2

        # N of GPUs
        n_GPUs = torch.cuda.device_count()

        # List of cuda device properties class
        # Minor bug fix here to not use just device 0 but whatever user specifies.
        x: list = [torch.cuda.get_device_properties(i) for i in range(n_GPUs)]
        print(
            f"Using CUDA device0 _CudaDeviceProperties(name={x[id].name}, "
            f"total_memory={x[id].total_memory / conversion_factor}MB)"
        )
        if n_GPUs > 0:
            # torch.cuda.set_device(0)  # OPTIONAL: Set GPU ID
            for index_GPU in range(1, n_GPUs):
                memory_MB = x[index_GPU].total_memory / conversion_factor
                print(
                    f"           device{index_GPU} _CudaDeviceProperties(name='{x[index_GPU].name}', "
                    f"total_memory={memory_MB}MB)"
                )

    return device

if __name__ == "__main__":
    _ = select_device(0)
