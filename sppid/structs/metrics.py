"""Data structures."""

from typing import Any, Dict, Iterable

from dataclasses import dataclass, field


@dataclass
class DCAMetrics:
     
    """Storage for DCA metrics.
    """
    
    ID: str
    id1: str = field(init=False)
    id2: str = field(init=False)
    seq_len: int
    chain_a_len: int
    chain_b_len: int
    msa1_depth: int
    msa2_depth: int
    msa_depth: int
    n_eff: int
    apc: bool
    mean: float
    median: float
    maximum: float
    minimum: float

    def __post_init__(self):
        self.id1, self.id2 = self.ID.split('-')


@dataclass
class RF2TMetrics:

    """Storage for RosettaFold 2-track metrics.
    """
    
    ID: str
    id1: str = field(init=False)
    id2: str = field(init=False)
    seq_len: int
    chain_a_len: int
    chain_b_len: int
    msa1_depth: int
    msa2_depth: int
    msa_depth: int
    n_eff: int
    mean: float
    median: float
    maximum: float
    minimum: float

    def __post_init__(self):
        self.id1, self.id2 = self.ID.split('-')


@dataclass
class ModelMetrics:
     
    """Storage for AlphaFold2 model metrics.
    """
    
    ID: str
    id1: str = field(init=False)
    id2: str = field(init=False)
    n_contacts: int
    mean_interfact_plddt: float
    pdockq: float

    def __post_init__(self):
        self.id1, self.id2 = self.ID.split('-')

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