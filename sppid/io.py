"""Tools for input and output."""

from typing import Any, Dict, Iterable, Tuple, Union

from csv import DictWriter
from dataclasses import asdict, dataclass, field, fields
from io import TextIOWrapper
import os
import shutil
import tempfile

from bioino import FastaCollection
from carabiner import print_err
from carabiner.cast import cast
import numpy as np
from pandas import DataFrame

from .structs import ModelMetrics
from .src_speedppi.alphafold.data import foldonly, pair_msas

@dataclass
class ATMRecord:

    """Parse an ATM record from a PDB line.

    """

    line: str
    name: str = field(init=False)
    atm_no: int = field(init=False)
    atm_name: str = field(init=False)
    atm_alt: str = field(init=False)
    res_name: str = field(init=False)
    chain: str = field(init=False)
    res_no: int = field(init=False)
    insert: str = field(init=False)
    resid: str = field(init=False)
    x: float = field(init=False)
    y: float = field(init=False)
    z: float = field(init=False)
    occ: float = field(init=False)
    B: float = field(init=False)

    def __post_init__(self):
        self.name = self.line[0:6].strip(),
        self.atm_no = int(self.line[6:11]),
        self.atm_name = self.line[12:16].strip(),
        self.atm_alt = self.line[17],
        self.res_name = self.line[17:20].strip(),
        self.chain = self.line[21],
        self.res_no = int(self.line[22:26]),
        self.insert = self.line[26].strip(),
        self.resid = self.line[22:29],
        self.x = float(self.line[30:38]),
        self.y = float(self.line[38:46]),
        self.z = float(self.line[46:54]),
        self.occ = float(self.line[54:60]),
        self.B = float(self.line[60:66])


def get_fasta_ids(file: Union[str, TextIOWrapper]):

    """Get IDs of FASTA sequences.

    """
    return list(f.name for f in FastaCollection.from_file(file).sequences)


def _copy_contents(source: TextIOWrapper, destination: TextIOWrapper):
    source.seek(0), destination.seek(0)
    for line in source:
        destination.write(str.encode(line))
    source.seek(0), destination.seek(0)
    return None


def load_msa_pair(msa_file1: Union[str, TextIOWrapper], 
                  msa_file2: Union[str, TextIOWrapper]) -> Tuple[Dict[str, Any], int]:

    """Load and join a pair of protein MSAs.

    """
    msa_file1, msa_file2 = (cast(f, to=TextIOWrapper) for f in (msa_file1, msa_file2))
    with tempfile.NamedTemporaryFile() as fp1, tempfile.NamedTemporaryFile() as fp2:
        _copy_contents(msa_file1, fp1), _copy_contents(msa_file2, fp2)
        id1, id2 = (get_fasta_ids(f.name)[0] for f in (fp1, fp2))
        if id1 == id2:
            print_err(f"Warning! Both queries have the same ID: {id1}")
        # Get features. The features are prefetched on CPU.
        #Pair and block MSAs
        (msa1, species1), (msa2, species2) = (pair_msas.read_a3m(f) for f in (fp1.name, fp2.name))
    seq1, seq2 = msa1[0], msa2[0]
    #Get the unique ox seqs from the MSAs
    (u_species1, idx1), (u_species2, idx2) = (np.unique(species, return_index=True) 
                                              for species in (species1, species2))
    #This is a list with seqs
    paired_msa = pair_msas.pair_msas(u_species1, u_species2, msa1[idx1], msa2[idx2])
    #Block the MSA
    gaps1, gaps2 = '-' * len(seq2), '-' * len(seq1)
    # The msas must be str representations of the blocked+paired MSAs here
    blocked_msa = [f"{seq}{gaps1}" for seq in msa1] + [f"{gaps2}{seq}" for seq in msa2]

    # The msas must be str representations of the blocked+paired MSAs here
    #Define the data pipeline
    feature_dict = foldonly.FoldDataPipeline().process(
        input_sequence=f"{seq1}{seq2}",
        input_description=":".join(sorted([id1, id2])),
        input_msas=[paired_msa, blocked_msa],
    )
    # Introduce chain breaks for oligomers
    feature_dict['residue_index'][len(seq1):] += 200
    feature_dict['ID'] = "-".join(sorted([id1, id2]))
    feature_dict['id1'] = id1
    feature_dict['id2'] = id2
    return feature_dict, len(seq1)



@dataclass
class Dataset:

    """
    
    """

    #Data
    dataset: str
    #First seq
    target_seq: str
    target_id: str
    indices: Iterable[int]
    msa_dir: str
    size: int = field(init=False)

    def __post_init__(self):
        if len(self.indices) < 5:
            self.indices = np.concatenate([self.indices] * 5)
        self.size = len(self.indices)

    def __len__(self):
        return self.size

    def __getitem__(self,
                    index: int) -> Dict[str, Any]:
        #Here the dataloading takes place
        index = self.indices[index] #This allows for loading more ex than indices
        row = self.dataset.loc[index]
        query_id = row['ID']
        msa_filenames = (os.path.join(self.msa_dir, f"{_id}.a3m") 
                         for _id in (self.target_id, query_id))
        feature_dict, target_length = load_msa_pair(*msa_filenames)
        return feature_dict


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


def write_metrics(metrics: Union[Iterable[ModelMetrics], ModelMetrics],
                  filename: str, 
                  mode: str = 'w'):
    
    """
    
    """

    if isinstance(metrics, ModelMetrics):
        metrics = [metrics]
    if not all(isinstance(m, ModelMetrics) for m in metrics):
        raise TypeError("All metrics must be ModelMetrics objects.")
    with open(filename, mode) as f:
        w = DictWriter(f, fieldnames=[_field.name for _field in fields(metrics[0])])
        if mode == 'w':
            w.writeheader()
        for metric in metrics:
            w.writerow(asdict(metric))

    return None
