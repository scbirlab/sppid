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

@dataclass
class MSAName:
    name: str
    input_name: str = field(init=False)
    prefix: str = field(init=False)
    gene_name: str = field(init=False)
    gene_and_species: str = field(init=False)

    def __post_init__(self):
        self.input_name = "".join(self.name)
        self.name = "".join(dropwhile(lambda s: s == ">", self.name)).rstrip()  # Strip out leading ">"
        try:
            self.prefix, self.gene_name, self.gene_and_species = self.name.split("|")
        except ValueError:
            print_err(self.name)
            self.prefix, self.gene_name, self.gene_and_species = None, None, None

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
        # pprint_dict(self.info)
        if "OX" in self.info:
            species_id = self.info["OX"]
        elif "TaxID" in self.info:
            species_id = self.info["TaxID"]
        else:
            species_id = '-1'
            print_err(f"MSA has no species info. Description string: {self.description}")
        try:
            self.species_id = int(species_id)
        except:
            self.species_id = -1
            
    def __str__(self):
        return self.description

@dataclass
class MSALine:
    name: str
    description: str
    sequence: str
    gene_and_species: str = field(init=False)
    gap_fraction: float = field(init=False)

    def __post_init__(self):
        self.sequence = ''.join(letter for letter in self.sequence if not letter.islower())  # remove insertions(?)
        self.name = MSAName(self.name)
        self.gene_and_species = self.name.gene_and_species
        self.description = MSADescription(self.description)
        self.gap_fraction = self.sequence.count('-') / float(len(self))

    def __len__(self):
        return len(self.sequence)

    def __repr__(self):
        return f"MSA line with name {self.name} and sequence length {len(self)}"

    def __str__(self):
        return f">{self.name} {self.description}\n{self.sequence}"


class PairedMSALine(MSALine):

    def __post_init__(self):
        if not " :: " in self.name:
            raise ValueError(f"Paired MSA must contain ' :: ' separator in name: {self.name}")
        self.name = tuple(MSAName(name) for name in self.name.split(' :: '))
        self.gene_and_species = '-'.join(name.gene_and_species for name in self.name)
        self.description = tuple(MSADescription(desc) for desc in self.description.split(' :: '))
        self.gap_fraction = self.sequence.count('-') / float(len(self))

    def __repr__(self):
        return "Paired " + super().__repr__(self)

    def __str__(self):
        return f">{' :: '.join(self.name)} {' :: '.join(self.description)}\n{self.sequence}"


@dataclass
class MSA:

    """MSA object which can be used for downstream analyses.
    """

    name: str = field(init=False)
    lines: Iterable[MSALine]
    sequence_labels: Iterable[str] = field(init=False)
    seq_length: int = field(init=False)
    sequence_token_ids: Iterable[int] = field(init=False)

    def __post_init__(self):
        self.lines = tuple(self.lines)
        self.name = self.lines[0].gene_and_species
        self.sequence_labels = tuple(line.name for line in self.lines)
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
        for _field in fields(self):
            original_items = getattr(self, _field.name) 
            if not isinstance(original_items, int):  # seq length
                setattr(self, _field.name, 
                        [item for i, item in enumerate(original_items) if i in indices])
        final_len = len(self)
        print_err(f"Filtered out {start_len - final_len}/{start_len} lines from MSA.")
        return self

    def filter_by_known_species(self):
        print_err("Filtering MSA by known species.")
        species_id = (line.description.species_id for line in self.lines)
        indices_to_keep = (i for i, _id in enumerate(species_id) if _id > -1)
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
        return f"MSA of sequence length {self.seq_length}, with {len(self)} sequences."

    def write(self, file=sys.stdout):
        for line in self.lines:
            print(line, file=file)


class PairedMSA(MSA):

    """Paired MSA object which can be used for DCA and AlphaFold2 analyses.
    """

    def __init__(self, 
                 chain_a_length: int, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chain_a_length = chain_a_length
        self.chain_b_length = self.seq_length - self.chain_a_length

    @staticmethod
    def join_msa(msa1: MSA, 
                 msa2: Optional[MSA] = None, 
                 blocked: bool = False):
        msa1 = msa1.filter_by_known_species()
        if msa2 is None:
            msa2 = deepcopy(msa1)
        else:
            msa2 = msa2.filter_by_known_species()
        #Get the matches
        species1, species2 = ([line.description.species_id for line in msa.lines]
                              for msa in (msa1, msa2))
        query_species = species1[0]
        common_species = set(species1).intersection(species2)  # Python `set.intersection` is much faster than np.intersect1d
        try:
            common_species.remove(query_species)
        except ValueError:
            raise ValueError(f"Query species {query_species} is not among the shared species in the MSAs:" 
                             + "\n" + '\n'.join(sorted(common_species)))
        common_species = [query_species] + sorted(common_species)
        #Go through all matching and select the first (top) hit
        msa_lines = [] 
        for species in common_species:
            idx1, idx2 = (sp.index(species) for sp in (species1, species2))
            msa_lines.append(PairedMSALine(name=" :: ".join(msa.lines[i].name.name for i, msa in zip((idx1, idx2), (msa1, msa2))),
                                           description=" :: ".join(msa.lines[i].description.description for i, msa in zip((idx1, idx2), (msa1, msa2))),
                                           sequence="".join(msa.lines[i].sequence for i, msa in zip((idx1, idx2), (msa1, msa2)))))
        if blocked:
            msa_lines += self.__make_blocked(msa1, msa2)
        return msa_lines, msa1.seq_length

    @staticmethod
    def __make_blocked(msa1: MSA, msa2: MSA):
        seq1, seq2 = (msa.lines[0].sequence for msa in (msa1, msa2))
        gaps1, gaps2 = ('-' * len(seq) for seq in (seq1, seq2))
        # The msas must be str representations of the blocked+paired MSAs here
        block1 = [PairedMSALine(name=f"{line.name} :: __BLOCK__", 
                                description=f"{line.description} :: __BLOCK__", 
                                sequence="".join(line.sequence, gaps1))
                  for line in msa1.lines]
        block2 = [PairedMSALine(name=f"__BLOCK__ :: {line.name}", 
                                description=f"__BLOCK__ :: {line.description}", 
                                sequence="".join(gaps2, line.sequence))
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