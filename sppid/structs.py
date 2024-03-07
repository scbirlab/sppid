"""Data structures."""

from dataclasses import dataclass, field

@dataclass
class ModelMetrics:
    ID: str
    id1: str = field(init=False)
    id2: str = field(init=False)
    n_contacts: int
    mean_interfact_plddt: float
    pdockq: float

    def __post_init__(self):
        self.id1, self.id2 = self.ID.split('-')