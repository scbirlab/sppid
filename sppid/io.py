"""Tools for input and output."""

from typing import Any, Dict, Iterable, Optional, Tuple, Union

from csv import DictWriter
from dataclasses import is_dataclass, asdict
from io import TextIOWrapper
import os
import shutil
import tempfile

from bioino import FastaCollection
from carabiner import print_err
from carabiner.cast import cast
import numpy as np
from pandas import DataFrame

from .structs.pdb import ATMRecord
from .structs.metrics import ModelMetrics
from .structs.msa import PairedMSA
from .src_speedppi.alphafold.data import foldonly, pair_msas

def load_msa_pair(msa_file1: Union[str, TextIOWrapper], 
                  msa_file2: Optional[Union[str, TextIOWrapper]] = None) -> Tuple[Dict[str, Union[str, int]], int]:

    """Load and join a pair of protein MSAs.

    """
    paired_msa = PairedMSA.from_file(msa_file1, msa_file2, blocked=True)
    msa_seqs = paired_msa.sequences()
    # The msas must be str representations of the blocked+paired MSAs here
    # Define the data pipeline
    ids = paired_msa.lines[0].description
    feature_dict = foldonly.FoldDataPipeline().process(
        input_sequence=msa_seqs[0],
        input_description=":".join(sorted(ids)),
        input_msas=msa_seqs,
    )
    # Introduce chain breaks for oligomers
    feature_dict['residue_index'][len(seq1):] += 200
    feature_dict['ID'] = "-".join(sorted(ids))
    feature_dict['id1'], feature_dict['id2'] = ids
    return feature_dict, paired_msa.chain_a_length


def save_design(pdb_info,
                output_name: str, 
                chainA_length: int) -> None:

    """Save the resulting protein-peptide design to a pdb file.

    """

    chain_name = 'A'
    with open(output_name, 'w') as f:
        pdb_contents = pdb_info.split('\n')
        for line in pdb_contents:
            try:
                record = ATMRecord(line)
                if record.res_no > chainA_length:
                    chain_name = 'B'
                outline = f"{line[:21]}{chain_name}{line[22:]}"
                f.writeline(outline)
            except Exception:
                f.writeline(line)
    return None


def write_metrics(metrics: Union[Iterable, Any],
                  filename: Union[str, TextIOWrapper], 
                  mode: str = 'w'):
    
    """
    
    """

    if is_dataclass(metrics):
        metrics = [metrics]
    if not all(is_dataclass(m) for m in metrics):
        raise TypeError("All metrics must be dataclass objects.")
    f = cast(filename, to=TextIOWrapper)
    w = DictWriter(f, fieldnames=list(asdict(metrics[0])), delimiter='\t')
    if mode == 'w':
        w.writeheader()
    for metric in metrics:
        w.writerow(asdict(metric))

    return None
