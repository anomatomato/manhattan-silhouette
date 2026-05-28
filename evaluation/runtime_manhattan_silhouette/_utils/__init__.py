"""
This file creates some utils for performing an experimental evaluation.
They are usually copy and paste stuff from other projects.
Keeping them in a separate file makes it easier to share them between projects.
"""

from .instance import make_instance
from .instance_db import InstanceDB
from .schema import InstanceKind, InstanceName, ManhattanSilhouetteInstance, ResultKey
from .strategies import STRATEGIES

__all__ = [
    "STRATEGIES",
    "InstanceDB",
    "InstanceName",
    "InstanceKind",
    "ManhattanSilhouetteInstance",
    "ResultKey",
    "make_instance",
]
