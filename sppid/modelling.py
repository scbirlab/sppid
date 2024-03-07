"""Tools for setting up and using models."""

from typing import Optional

import os
from time import time

from .src_speedppi.alphafold.model import config, data, model

_params_path = os.path.join(os.path.dirname(__file__), "data")

def make_model_runner(num_ensemble: int = 1,
                      max_recycles: int = 10,
                      param_dir: Optional[str] = None,
                      model_name: Optional[str] = None):
    
    """Generate an AlphaFold2 model runner.

    """

    if model_name is None:
        model_name = 'model_1'
    if param_dir is None:
        param_dir = _params_path

    model_config = config.model_config(model_name)
    model_config.data.eval.num_ensemble = num_ensemble
    model_config.data.common.num_recycle = max_recycles
    model_config.model.num_recycle = max_recycles

    model_params = data.get_model_haiku_params(model_name=model_name, 
                                               data_dir=param_dir)

    return model.RunModel(model_config, model_params)
