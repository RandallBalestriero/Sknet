#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow import constant
import numpy as np

__all__ = [
        "dataset",
        "layers",
        "ops",
        "utils",
        "losses",
        "schedules",
        "optimizers",
        "networks"]

__version__ = 'alpha.1'

from .base import *
from . import *
from .dataset import Dataset
from .networks import Network
#from .optimize import loss as losses
from .optimize import optimizer as optimizers
from .optimize import schedule as schedules





