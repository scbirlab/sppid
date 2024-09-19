"""Data structures for multiple sequence alignments."""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from copy import deepcopy
from io import TextIOWrapper
from itertools import dropwhile
import sys

from dataclasses import asdict, dataclass, field, fields

from bioino import FastaCollection
from carabiner import pprint_dict, print_err
from carabiner.cast import cast
from carabiner.itertools import batched
from tqdm.auto import tqdm

_A3M_ALPHABET = tuple("ARNDCQEGHILKMFPSTWYV-")
_A3M_ALPHABET_DICT = dict(zip(_A3M_ALPHABET, range(len(_A3M_ALPHABET))))
_PAIRED_SPACER = ':::'

@dataclass
class MSAName:
    name: str
    input_name: str = field(init=False)
    database: str = field(init=False)
    unique_id: str = field(init=False)
    entry_name: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.name, str):
            try:
                self.input_name = "".join(self.name)
            except TypeError:
                raise TypeError(f"MSA name `{self.name}` is type {type(self.name)}.")
        else:
            self.input_name = self.name
        self.name = "".join(dropwhile(lambda s: s == ">", self.name)).rstrip()  # Strip out leading ">"
        try:
            self.database, self.unique_id, self.entry_name = self.name.split("|")
        except ValueError:
            print_err(self.name)
            self.database, self.unique_id, self.entry_name = "__NO_NAME__", "__NO_ENTRY_ID__", "__NO_ENTRY_NAME__"

    def __str__(self):
        return self.name


@dataclass
class MSADescription:
    description: str
    species_id: str = field(init=False)
    prefix: str = field(init=False)
    info: Mapping[str, Union[int, str]] = field(init=False)

    def __post_init__(self):
        desc_esc_eq = ":eq:".join(self.description.split(" = "))
        split_on_eq = [item.strip() for item in desc_esc_eq.split("=")]
        prefix_split_on_space = split_on_eq[0].split()
        if len(prefix_split_on_space) > 1:
            self.prefix = " ".join(prefix_split_on_space[:-1])
            split_on_eq[0] = prefix_split_on_space[-1]
        else:
            self.prefix = None
        info = {}
        for i in range(1, len(split_on_eq)):
            key = split_on_eq[i - 1].split()[-1].strip()
            val = ' '.join(split_on_eq[i].split()[:-1]).strip()
            if val.isdigit():
                val = int(val)
            info[key] = val
        self.info = info
        if "OX" in self.info:  # NCBI identifier. Doesn't exist for everything
            species_id = f"NCBI:{self.info['OX']}"
        elif "TaxID" in self.info:
            species_id = f"TaxID:{self.info['TaxID']}"
        elif "OS" in self.info:  # UniProt species name fallback
            species_id = f"Name:{self.info['OS']}"
        else:
            species_id = -1
            if self.description != '__BLOCK_GAPS__':
                print_err(f"MSA has no species info. Description string: {self.description.rstrip()}")
        self.species_id = species_id
            
    def __str__(self):
        return self.description


@dataclass
class MSALine:
    name: str
    description: str
    sequence: str
    entry_name: str = field(init=False)
    gap_fraction: float = field(init=False)

    def __post_init__(self):
        self.sequence = ''.join(letter for letter in self.sequence if not letter.islower())  # remove insertions(?)
        self.name = MSAName(self.name)
        self.unique_id = self.name.unique_id
        self.entry_name = self.name.entry_name
        self.description = MSADescription(self.description)
        self.gap_fraction = self.sequence.count('-') / float(len(self))

    def __len__(self):
        return len(self.sequence)

    def __repr__(self):
        return f"MSALine(name='{self.name}', length={len(self)})"

    def __str__(self):
        return f">{str(self.name)} {str(self.description)}\n{self.sequence}"


class PairedMSALine(MSALine):

    def __post_init__(self):
        if not _PAIRED_SPACER in self.name:
            raise ValueError(f"Paired MSA must contain '{_PAIRED_SPACER}' separator in name: {self.name}")
        self.name = tuple(MSAName(name) for name in self.name.split(_PAIRED_SPACER))
        self.unique_id = '-'.join(name.unique_id for name in self.name)
        self.entry_name = '-'.join(name.entry_name for name in self.name)
        self.description = tuple(MSADescription(desc) for desc in self.description.split(_PAIRED_SPACER))
        self.gap_fraction = self.sequence.count('-') / float(len(self))

    def __repr__(self):
        return "Paired " + super().__repr__(self)

    def __str__(self):
        return f">{_PAIRED_SPACER.join(map(str, self.name))} {_PAIRED_SPACER.join(map(str, self.description))}\n{self.sequence}"


@dataclass
class MSA:

    """MSA object which can be used for downstream analyses.
    """

    lines: Iterable[MSALine]
    name: str = field(init=False)
    sequence_labels: Iterable[str] = field(init=False)
    seq_length: int = field(init=False)
    sequence_token_ids: Iterable[int] = field(init=False)

    def __post_init__(self):
        self.lines = tuple(self.lines)
        self.name = self.lines[0].unique_id
        self.sequence_labels = tuple(line.unique_id for line in self.lines)
        seq_lengths = [len(line) for line in self.lines]
        seq_length = set(seq_lengths)
        if len(seq_length) > 1:
            raise AttributeError(f"Sequence lengths are not uniform! Found lengths: {', '.join(map(str, seq_length))}")
        if len(seq_length) == 0:
            raise AttributeError(f"No sequences! The lines are: {', '.join(map(str, self.lines))}")
        self.seq_length = seq_length.pop()
        self.sequence_token_ids = [[_A3M_ALPHABET_DICT.get(letter, len(_A3M_ALPHABET_DICT) - 1) 
                                     for letter in line.sequence]
                                    for line in self.lines]

    def sequences(self) -> List[str]:
        return [line.sequence for line in self.lines]

    def gap_fraction(self) -> List[float]:
        return [line.gap_fraction for line in self.lines]

    @classmethod
    def from_file(cls, file: Union[str, TextIOWrapper]):
        collection = list(FastaCollection.from_file(file).sequences)
        # print(collection[0])
        return cls(MSALine(**asdict(seq)) for seq in tqdm(collection))

    def __len__(self):
        return len(self.lines)

    def neff(self, identity_threshold=.62):
        """Calculate the number of effective sequences.
        """
        remaining_msa = deepcopy(self.sequence_token_ids)
        threshold = float(self.seq_length * identity_threshold)
        n_effective = 0
        while len(remaining_msa) > 0:
            print_err(f"\rClustering at identity threshold {identity_threshold} ({threshold}/{self.seq_length} "
                      f"positions): Neff = {n_effective} | remaining to cluster: {len(remaining_msa)}", 
                      end='')
            first_remaining_msa = remaining_msa[0]
            msa_diff = [sum((other - b) != 0 for other, b in zip(row, first_remaining_msa))
                        for row in remaining_msa]
            remaining_msa = [line for diff, line in zip(msa_diff, remaining_msa) 
                             if diff > threshold]
            n_effective += 1
        print_err()
        return n_effective

    def _filter_by_index(self, indices=Iterable[int]):
        start_len = len(self)
        indices = set(indices)
        new_copy = deepcopy(self)
        for _field in fields(self):
            if _field.name not in ('name', 'seq_length'):
                original_items = getattr(self, _field.name) 
                if not isinstance(original_items, int):  # seq length
                    setattr(new_copy, _field.name, 
                            [item for i, item in enumerate(original_items) if i in indices])
        final_len = len(new_copy)
        print_err(f"Filtered out {start_len - final_len}/{start_len} lines from MSA.")
        return new_copy

    def filter_by_known_species(self):
        print_err("Filtering MSA by known species.")
        species_id = (line.description.species_id for line in self.lines)
        indices_to_keep = (i for i, _id in enumerate(species_id) if _id != -1)
        return self._filter_by_index(indices_to_keep)

    def filter_by_gap_fraction(self, max_gap_fraction=1.):
        if max_gap_fraction < 1.:
            print_err(f"Filtering MSA by gap fraction < {max_gap_fraction}.")
            indices_to_keep = (i for i, line in enumerate(self.lines)
                               if line.gap_fraction <= max_gap_fraction)
            return self._filter_by_index(indices_to_keep)
        else:
            return self

    def __str__(self):
        return f"MSA(name={self.name}) of sequence length {self.seq_length}, with {len(self)} sequences."

    def write(self, file=sys.stdout):
        for line in self.lines:
            print(line, file=file)


class PairedMSA(MSA):

    """Paired MSA object which can be used for co-evolutionary analyses.
    """

    def __init__(self, 
                 chain_a_length: int, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chain_a_length = chain_a_length
        self.chain_b_length = self.seq_length - self.chain_a_length

    @staticmethod
    def _check_ref_match(msa1: MSA, 
                         msa2: MSA) -> None:
        id1, id2 = msa1.lines[0].description.species_id, msa2.lines[0].description.species_id
        if id1 != id2:
            raise AttributeError(f"MSA reference species do not match: {id1}, {id2}")
        if id1 == -1 or id2 == -1:
            raise ValueError(f"At least one MSA reference species is unknown: {id1}, {id2}")
        return None

    @staticmethod
    def join_msa(msa1: MSA, 
                 msa2: Optional[MSA] = None, 
                 blocked: bool = False):
       
        if msa2 is None:
            msa2 = deepcopy(msa1)

        PairedMSA._check_ref_match(msa1, msa2)
        msa1, msa2 = (msa.filter_by_known_species() for msa in (msa1, msa2))
        PairedMSA._check_ref_match(msa1, msa2)
        #Get the matches
        species1, species2 = ([line.description.species_id for line in msa.lines]
                               for msa in (msa1, msa2))
        query_species = species1[0]
        common_species = set(species1).intersection(species2)  # Python `set.intersection` is much faster than np.intersect1d
        try:
            common_species.remove(query_species)
        except KeyError:
            raise KeyError(f"Query species {query_species} is not among the shared species in the MSAs:" 
                            + "\n" + '\n'.join(sorted(common_species)))
        common_species = [query_species] + sorted(common_species)
        #Go through all matching and select the first (top) hit
        msa_lines = [] 
        for species in common_species:
            idx1, idx2 = (sp.index(species) for sp in (species1, species2))
            msa_lines.append(PairedMSALine(name=_PAIRED_SPACER.join(str(msa.lines[i].name) for i, msa in zip((idx1, idx2), (msa1, msa2))),
                                           description=_PAIRED_SPACER.join(str(msa.lines[i].description) for i, msa in zip((idx1, idx2), (msa1, msa2))),
                                           sequence="".join(msa.lines[i].sequence for i, msa in zip((idx1, idx2), (msa1, msa2)))))

        if blocked:  # make blocked MSA for the individual proteins not belonging to a species pair
            idx1, idx2 = ([i for i, species in enumerate(sp) if species not in common_species] 
                          for sp in (species1, species2))
            msa1, msa2 = (msa._filter_by_index(idx) for idx, msa in zip((idx1, idx2), (msa1, msa2)))
            msa_lines += PairedMSA.__make_blocked(msa1, msa2)
        
        return msa_lines, msa1.seq_length

    @staticmethod
    def __make_blocked(msa1: MSA, msa2: MSA, gap_char: str = '-'):
        gaps1, gaps2 = (gap_char * msa.seq_length for msa in (msa1, msa2))
        # The msas must be str representations of the blocked+paired MSAs here
        block1 = [PairedMSALine(name=f"{line.name}{_PAIRED_SPACER}xx|__BLOCK_GAPS__|__BLOCK_GAPS__", 
                                description=f"{line.description}{_PAIRED_SPACER}__BLOCK_GAPS__", 
                                sequence="".join([line.sequence, gaps2]))
                  for line in msa1.lines]
        block2 = [PairedMSALine(name=f"xx|__BLOCK_GAPS__|__BLOCK_GAPS__{_PAIRED_SPACER}{line.name}", 
                                description=f"__BLOCK_GAPS__{_PAIRED_SPACER}{line.description}", 
                                sequence="".join([gaps1, line.sequence]))
                  for line in msa2.lines]
        return block1 + block2

    @classmethod
    def from_msa(cls, 
                 msa1: MSA, 
                 msa2: Optional[MSA] = None,
                 blocked: bool = False):
        msa_lines, chain_a_length = cls.join_msa(msa1, msa2, blocked=blocked)
        return cls(lines=msa_lines, chain_a_length=chain_a_length)

    @classmethod
    def from_file(cls, 
                  file1: Union[str, TextIOWrapper],
                  file2: Optional[Union[str, TextIOWrapper]] = None,
                  blocked: bool = False):
        """Read A3M file(s).

        """
        msa1 = MSA.from_file(file1)
        if file2 is None:
            msa2 = deepcopy(msa1)
        else:
            msa2 = MSA.from_file(file2)
        return cls.from_msa(msa1, msa2, blocked=blocked)

    def __str__(self):
        return "Paired " + super().__str__()