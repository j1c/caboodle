import os
import json
import time
from pathlib import Path
import numpy as np
import torch


def check_mps() -> bool:
    """
    Helper function for checking if MPS backend is available for M chip macs.

    Returns
    -------
    avail : bool
        True if MPS is available. False, otherwise.
    """

    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )
        return False
    else:
        return True


def setup_device(cuda, seed, local_rank=0):
    # Set up device
    np.random.seed(seed)
    device = "cpu"
    torch.manual_seed(seed)
    if cuda:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            device = f"cuda:{local_rank}"
        else:
            raise RuntimeError("Attempted to use CUDA but not available!")

    return device


def summary(model) -> None:
    """Prints summary of model parameters

    Parameters
    ----------
    model : nn.Module
        Pytorch model
    """
    # Count top_level params
    _num_params = lambda m: sum(p.numel() for p in m.parameters())

    print(f"Model created with {_num_params(model)} parameters")
    for child in model.children():
        name = child.__class__.__name__
        print(f"\t{name} has {_num_params(child)} parameters")


def setup_paths(args):
    # Setup an output path and log files
    if args.output_folder is not None:
        output_folder = os.path.join(args.path_output, args.output_folder)
        if not output_folder.endswith("/"):
            output_folder += "/"
    else:
        output_folder = args.path_output + time.strftime("%Y%m%d-%I%M%S%p/", time.localtime())
    output_folders = {}
    for subfolder in ["", "models", "plots"]:
        path = output_folder + subfolder + "/"
        output_folders[subfolder] = path
        if not os.path.exists(path):
            Path(path).mkdir(exist_ok=True, parents=True)
        if not args.multigpu:
            with open(output_folder + "args.json", "w") as f:
                json.dump(vars(args), f)
        else:
            if torch.distributed.get_rank() == 0:
                with open(output_folder + "args.json", "w") as f:
                    json.dump(vars(args), f)
    with open(output_folder + "args.json", "w") as f:
        json.dump(vars(args), f)
    return output_folders
