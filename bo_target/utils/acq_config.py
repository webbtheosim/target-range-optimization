"""Acquisition function dispatch: string name -> acquisition class."""

from bo_target.acquisitions import EI, HV, LCB, TB, BAX, RS

_registry = {
    "tb": TB,
    "hv": HV,
    "ei": EI,
    "lcb": LCB,
    "bax": BAX,
    "rs": RS,
}


def acq_config(acquisition_name):
    """Return the acquisition class for a given short name (e.g. ``"tb"``)."""
    if acquisition_name not in _registry:
        raise ValueError(f"Unknown acquisition: {acquisition_name}")
    return _registry[acquisition_name]
