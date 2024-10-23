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
