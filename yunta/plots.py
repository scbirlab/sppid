"""Plotting functions."""

from typing import Optional

import os

from carabiner import colorblind_palette, print_err
from carabiner.mpl import grid
from pandas import DataFrame
import numpy as np

def plot_matrix(m: np.ndarray, 
                filename_prefix: Optional[str] = None,
                format: str = 'png',
                dpi: int = 300):
    
    fig, axes = grid()
    im = axes.imshow(m, cmap='magma', vmin=0.)
    fig.colorbar(im, shrink=.6)

    if filename_prefix is not None:
        plot_dir = os.path.dirname(filename_prefix)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        data_file = f"{filename_prefix}.tsv"
        print_err(f"Saving matrix data as {data_file}")
        (DataFrame(m, 
                   index=np.arange(m.shape[0]), 
                   columns=np.arange(m.shape[1]))
         .to_csv(data_file, sep='\t', index=False))
        plot_file = f"{filename_prefix}.{format}"
        print_err(f"Saving matrix plot as {plot_file}")
        fig.savefig(plot_file, bbox_inches="tight", dpi=dpi)

    return fig, axes

def plot_dca(dca: np.ndarray, 
             filename_prefix: Optional[str] = None,
             format: str = 'png',
             dpi: int = 300):
    
    if filename_prefix is not None:
        filename_prefix += "_dca"

    return _plot_matrix(dca, filename_prefix=filename_prefix, format=format, dpi=dpi)