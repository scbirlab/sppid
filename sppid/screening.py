"""Tools to screen for PPIs."""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from csv import writer
from io import TextIOWrapper
import os
import random
import sys
from time import time

from carabiner import print_err, pprint_dict
from carabiner.cast import cast
from tqdm.auto import tqdm

from .io import load_msa_pair
from .modelling import make_model_runner
from .scoring import score_ppi
from .structs import ModelMetrics

def model_protein_interaction(msa_file1: Union[str, TextIOWrapper], 
                              msa_file2: Union[str, TextIOWrapper],
                              model_runner: Optional = None,
                              seed: Optional[int] = None,
                              *args, **kwargs):
    
    """Model a single PPI using a pair of MSA files.
    
    """

    if seed is None:
        seed = random.randrange(sys.maxsize)
    if model_runner is None:
        model_runner = make_model_runner(*args, **kwargs)

    feature_dict, chain_a_length = load_msa_pair(msa_file1, msa_file2)
    print_err(f"Modelling pair {feature_dict['ID']}...")
    # pprint_dict(feature_dict, message="Modelling pair")
    # Run the model - on GPU
    t0 = time()
    processed_feature_dict = model_runner.process_features(feature_dict, 
                                                           random_seed=seed)
    prediction_result = model_runner.predict(processed_feature_dict)
    print_err(f"It took {time() - t0} s to predict the interaction.")
    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt_b_factors = np.repeat(prediction_result['plddt'][:, np.newaxis], 
                                residue_constants.atom_type_num, 
                                axis=-1)
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
    )
    return feature_dict, unrelaxed_protein, chain_a_length


def evaluate_and_save_model(feature_dict: Mapping[str, Any],
                            unrelaxed_protein,  
                            chain_a_length: int,
                            filename: str,
                            pdockq_t: float = .5,
                            force_save: bool = False) -> ModelMetrics:

    """Evalulate a model and save PDB.
    
    """
    #Score - calculate the pDockQ (number of interface residues and average interface plDDT)
    pdockq, avg_interface_plddt, n_interface_contacts = score_ppi(unrelaxed_protein, chain_a_length)

    #Save if pDockQ > t
    if force_save or pdockq > pdockq_t:
        print_err(f"Saving {feature_dict['ID']} structure with pDockQ = {pdockq:.2f} as {filename}.")
        pdb, _ = protein.to_pdb(unrelaxed_protein)
        save_design(pdb, output_name, chain_a_length)
    else:
        print_err(f"Skipping saving {feature_dict['ID']} structure with pDockQ = {pdockq:.2f}.")
    
    return ModelMetrics(ID=feature_dict['ID'], n_contacts=n_interface_contacts, 
                        avg_interface_plddt=avg_interface_plddt, pdockq=pdockq)


def build_evaluate_and_save_model(msa_file1: Union[str, TextIOWrapper],
                                  output_dir: str,
                                  msa_file2: Optional[Union[str, TextIOWrapper]] = None,
                                  pdockq_t: float = .5,
                                  force_save: bool = True,
                                  seed: Optional[int] = None,
                                  model_runner: Optional = None,
                                  *args, **kwargs) -> ModelMetrics:

    """Predict the structure of a pair of proteins based on provided MSAs.

    """

    if msa_file2 is None:
        msa_file2 = msa_file1

    feature_dict, unrelaxed_protein, chain_a_length = model_protein_interaction(
        msa_file1=msa_file1, 
        msa_file2=msa_file2,
        seed=seed,
        model_runner=model_runner,
        *args, **kwargs
    )

    metric = evaluate_and_save_model(
        feature_dict=feature_dict,
        unrelaxed_protein=unrelaxed_protein,  
        chain_a_length=chain_a_length,
        filename=os.path.join(output_dir, f"{feature_dict['ID']}.pdb"),
        pdockq_t=pdockq_t,
        force_save=force_save,
    )

    return metric


def one_vs_many(msa_file1: Union[str, TextIOWrapper],
                msa_files2: Iterable[Union[str, TextIOWrapper]],
                output_dir: str,
                pdockq_t: float = .5,
                force_save: bool = False,
                seed: Optional[int] = None,
                model_runner: Optional = None,
                *args, **kwargs) -> List[ModelMetrics]:

    if not os.path.exits(output_dir):
        os.path.makedirs(output_dir)

    model_runner = make_model_runner(*args, **kwargs)
    msa_file1 = cast(msa_file1, to=str)
    msa_files2 = list(msa_files2)
    metrics = []
    print_err(f"Running {msa_file1} against {len(msa_files2)} MSAs...")
    for msa_file2 in tqdm(msa_files2):
        metrics.append(
            build_evaluate_and_save_model(
                msa_file1=msa_file1,
                msa_file2=cast(msa_file2, to=str),
                output_dir=output_dir,
                pdockq_t=pdockq_t,
                force_save=force_save,
                seed=seed,
                model_runner=model_runner,
            )
        )

    return metrics
    

def many_vs_many(msa_files1: Iterable[Union[str, TextIOWrapper]],
                 output_dir: str,
                 msa_files2: Optional[Iterable[Union[str, TextIOWrapper]]] = None,
                 pdockq_t: float = .5,
                 force_save: bool = True,
                 seed: Optional[int] = None,
                 model_runner: Optional = None,
                 *args, **kwargs) -> List[ModelMetrics]:

    model_runner = make_model_runner(*args, **kwargs)
    metrics = []
    
    msa_files1 = list(msa_files1)
    if msa_files2 is None:
        print_err("No second set of MSAs provided,"
                  " so screening all pairwise interactions from the first set.")
        msa_files2 = [f for f in msa_files1]
    print_err(f"Screening {len(msa_files1)} MSAs against {len(msa_files2)} MSAs...")
    for msa_file1 in tqdm(msa_files1):
        metrics += one_vs_many(
                msa_file1=msa_file1,
                msa_file2=cast(msa_file2, to=str),
                output_dir=output_dir,
                pdockq_t=pdockq_t,
                force_save=force_save,
                seed=seed,
                model_runner=model_runner,
            )
    return metrics