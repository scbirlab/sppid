"""Tools to screen for PPIs."""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from copy import deepcopy
from csv import writer
from io import TextIOWrapper
import os
import random
import sys
from time import time

from carabiner import print_err
from carabiner.cast import cast
import numpy as np
from tqdm.auto import tqdm

from .io import save_design
from .modelling import make_model_runner
from .scoring import score_ppi
from .src_speedppi.alphafold import protein, residue_constants
from .src_speedppi.alphafold.data import foldonly
from .structs.metrics import DCAMetrics, ModelMetrics, RF2TMetrics
from .structs.msa import MSA, PairedMSA

def _pair_msas(msa1: MSA, 
               msa2: Optional[MSA] = None,
               max_gap_fraction: float = 1.,
               blocked: bool = False) -> Mapping[str, Union[int, float]]:
    if msa2 is None:
        msa2 = deepcopy(msa1)
    print_err(msa1, msa2)
    return PairedMSA.from_msa(msa1, msa2, blocked=blocked).filter_by_gap_fraction(max_gap_fraction)


def _get_af2_features(paired_msa: PairedMSA) -> Dict[str, Union[str, int]]:

    msa_seqs = paired_msa.sequences()
    # The msas must be str representations of the blocked+paired MSAs here
    # Define the data pipeline
    ids = list(getattr(name, "unique_id") for name in paired_msa.lines[0].name)
    feature_dict = foldonly.FoldDataPipeline().process(
        input_sequence=msa_seqs[0],  # reference
        input_description="_".join(sorted(ids)),
        input_msas=[msa_seqs],
    )
    # Introduce chain breaks for oligomers
    feature_dict['residue_index'][paired_msa.chain_a_length:] += 200
    feature_dict['ID'] = "-".join(sorted(ids))
    feature_dict['uniprot_id_1'], feature_dict['uniprot_id_2'] = ids
    return feature_dict


def rf2track(msa1: MSA, 
             msa2: Optional[MSA] = None,
             cpu: bool = True,
             max_gap_fraction: float = .9,
             model: Optional = None) -> Tuple[np.ndarray, np.ndarray, RF2TMetrics]:
    paired_msa = _pair_msas(msa1, msa2, max_gap_fraction=max_gap_fraction, blocked=True)
    print_err(paired_msa)
    chain_a_length = paired_msa.chain_a_length
    chain_b_length = paired_msa.seq_length - paired_msa.chain_a_length
    neff = paired_msa.neff()

    if model is None:
        from rf2t_micro.predict_msa import Predictor
        import torch
        torch.cuda.empty_cache()
        model = Predictor(use_cpu=cpu)

    result, cα_coords = model.predict(np.asarray(paired_msa.sequence_token_ids), 
                                      chain_a_length=paired_msa.chain_a_length)

    result_interaction = result[:chain_a_length, chain_a_length:]
    metrics = RF2TMetrics(
        ID=paired_msa.name, 
        seq_len=paired_msa.seq_length,
        chain_a_len=paired_msa.chain_a_length,
        chain_b_len=paired_msa.chain_b_length,
        msa1_depth=len(msa1),
        msa2_depth=len(msa2) if msa2 is not None else len(msa1),
        msa_depth=len(paired_msa),
        n_eff=neff,
        maximum=np.max(result_interaction), 
        minimum=np.min(result_interaction), 
        mean=np.mean(result_interaction), 
        median=np.median(result_interaction)
    )
    return result, result_interaction, metrics


def rf2track_one_vs_many(msa_file1: Union[str, TextIOWrapper],
                         msa_file2: Optional[Iterable[Union[str, TextIOWrapper]]] = None,
                         cpu: bool = True) -> List[Tuple[np.ndarray, np.ndarray, RF2TMetrics]]:

    if msa_file2 is None:
        msa_file2 = [None]
    if isinstance(msa_file2, str) or isinstance(msa_file2, TextIOWrapper):
        msa_file2 = [msa_file2]
    msa1 = MSA.from_file(msa_file1)
    results = []
    print_err(f"Calculating contact matrix for {msa_file1} against {len(msa_file2)} MSAs...")
    from rf2t_micro.predict_msa import Predictor
    import torch
    torch.cuda.empty_cache()
    model = Predictor(use_cpu=cpu)
    for msa2 in tqdm(msa_file2):
        if msa2 is not None:
            msa2 = MSA.from_file(msa2)
        results.append(
            rf2track(
                msa1=msa1,
                msa2=msa2,
                model=model,
            )
        )

    return results


def paired_dca(msa1: MSA, 
               msa2: Optional[MSA] = None,
               apc: bool = False,
               max_gap_fraction: float = .9) -> Tuple[np.ndarray, np.ndarray, DCAMetrics]:

    paired_msa = _pair_msas(msa1, msa2, max_gap_fraction=max_gap_fraction)
    print_err(paired_msa)
    neff = paired_msa.neff()

    from .dca_torch import calculate_dca

    result = calculate_dca(msa=paired_msa, apc=apc)
    result_interaction = result[:paired_msa.chain_a_length, paired_msa.chain_a_length:]

    metrics = DCAMetrics(
        ID=paired_msa.name, 
        seq_len=paired_msa.seq_length,
        chain_a_len=paired_msa.chain_a_length,
        chain_b_len=paired_msa.chain_b_length,
        msa1_depth=len(msa1),
        msa2_depth=len(msa2) if msa2 is not None else len(msa1),
        msa_depth=len(paired_msa),
        n_eff=neff,
        apc=apc,
        maximum=np.max(result_interaction), 
        minimum=np.min(result_interaction), 
        mean=np.mean(result_interaction), 
        median=np.median(result_interaction)
    )

    return result, result_interaction, metrics


def dca_one_vs_many(msa_file1: Union[str, TextIOWrapper],
                    msa_file2: Optional[Iterable[Union[str, TextIOWrapper]]] = None,
                    apc: bool = False) -> List[DCAMetrics]:

    if msa_file2 is None:
        msa_file2 = [None]
    if isinstance(msa_file2, str) or isinstance(msa_file2, TextIOWrapper):
        msa_file2 = [msa_file2]
    msa1 = MSA.from_file(msa_file1)
    results = []
    print_err(f"Calculating DCA for {msa_file1} against {len(msa_file2)} MSAs...")
    for msa2 in tqdm(msa_file2):
        if msa2 is not None:
            msa2 = MSA.from_file(msa2)
        results.append(
            paired_dca(
                msa1=msa1,
                msa2=msa2,
                apc=apc,
            )
        )

    return results


def dca_many_vs_many(msa_files1: Iterable[Union[str, TextIOWrapper]],
                     msa_files2: Optional[Iterable[Union[str, TextIOWrapper]]] = None,
                     apc: bool = False) -> List[Tuple[np.ndarray, np.ndarray, DCAMetrics]]:
    results = []
    # msa_files1 = cast(msa_files1, to=lost)
    if msa_files2 is None:
        print_err("No second set of MSAs provided,"
                  " so screening all pairwise interactions from the first set.")
        msa_files2 = [f for f in msa_files1]
    print_err(f"Screening {len(msa_files1)} MSAs against {len(msa_files2)} MSAs...")
    for msa_file1 in tqdm(msa_files1):
        results += dca_one_vs_many(
                msa_file1=msa_file1,
                msa_file2=msa_files2,
                apc=apc,
            )
    return results
    

def model_protein_interaction(msa1: MSA, 
                              msa2: Optional[MSA] = None,
                              model_runner: Optional = None,
                              seed: Optional[int] = None,
                              max_gap_fraction: float = .9,
                              *args, **kwargs) -> Tuple[Mapping[str, Union[float, int]], Any, PairedMSA]:
    
    """Model a single PPI using a pair of MSA files.
    
    """
    if seed is None:
        seed = random.randrange(sys.maxsize)
    if model_runner is None:
        model_runner = make_model_runner(*args, **kwargs)

    paired_msa = _pair_msas(msa1, msa2, max_gap_fraction=max_gap_fraction, blocked=True)
    print_err(paired_msa)

    feature_dict = _get_af2_features(paired_msa)
    print_err(f"Modelling pair {feature_dict['ID']}...")
    # Run the model - on GPU
    t0 = time()
    #TODO: Swap the AlphaFold2 protein modelling for OpenFold (faster? PyTorch, open source)
    processed_feature_dict = model_runner.process_features(feature_dict, 
                                                           random_seed=seed)
    prediction_result = model_runner.predict(processed_feature_dict)
    print_err(f"It took {time() - t0} s to predict the interaction.")
    
    return paired_msa, feature_dict, processed_feature_dict, prediction_result


def evaluate_and_save_model(feature_dict,
                            processed_feature_dict: Mapping[str, Any],
                            prediction_result: Mapping[str, Any],  
                            chain_a_length: int,
                            filename: str,
                            pdockq_t: float = .5,
                            force_save: bool = False) -> ModelMetrics:

    """Evalulate a model and save PDB.
    
    """
    plddt_b_factors = np.repeat(prediction_result['plddt'][:, np.newaxis], 
                                residue_constants.atom_type_num, 
                                axis=-1)
    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
    )

    #Score - calculate the pDockQ (number of interface residues and average interface plDDT)
    pdockq, avg_interface_plddt, n_interface_contacts = score_ppi(
        unrelaxed_protein, 
        prediction_result['plddt'], 
        chain_a_length,
    )

    #Save if pDockQ > t
    if force_save or pdockq > pdockq_t:
        print_err(f"Saving {feature_dict['ID']} structure with pDockQ = {pdockq:.2f} as {filename}.")
        pdb, _ = protein.to_pdb(unrelaxed_protein)
        save_design(pdb, filename, chain_a_length)
    else:
        print_err(f"Skipping saving {feature_dict['ID']} structure with pDockQ = {pdockq:.2f}.")
    
    return ModelMetrics(ID=feature_dict['ID'], n_contacts=n_interface_contacts, 
                        mean_interfact_plddt=avg_interface_plddt, pdockq=pdockq)


def build_evaluate_and_save_model(msa1: MSA,
                                  output_dir: str,
                                  msa2: Optional[MSA] = None,
                                  pdockq_t: float = .5,
                                  force_save: bool = True,
                                  seed: Optional[int] = None,
                                  model_runner: Optional = None,
                                  *args, **kwargs) -> ModelMetrics:

    """Predict the structure of a pair of proteins based on provided MSAs.

    """
    paired_msa, feature_dict, processed_feature_dict, prediction_result = model_protein_interaction(
        msa1=msa1, 
        msa2=msa2,
        seed=seed,
        model_runner=model_runner,
        *args, **kwargs
    )

    metric = evaluate_and_save_model(
        feature_dict=feature_dict,
        processed_feature_dict=processed_feature_dict,
        prediction_result=prediction_result,  
        chain_a_length=paired_msa.chain_a_length,
        filename=os.path.join(output_dir, f"{feature_dict['ID']}.pdb"),
        pdockq_t=pdockq_t,
        force_save=force_save,
    )

    return metric


def model_one_vs_many(msa_file1: Union[str, TextIOWrapper],
                      output_dir: str,
                      msa_file2: Optional[Iterable[Union[str, TextIOWrapper]]] = None,
                      pdockq_t: float = .5,
                      force_save: bool = False,
                      seed: Optional[int] = None,
                      model_runner: Optional = None,
                      *args, **kwargs) -> List[ModelMetrics]:

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_runner = make_model_runner(*args, **kwargs)
    if msa_file2 is None:
        msa_file2 = [None]
    if isinstance(msa_file2, str) or isinstance(msa_file2, TextIOWrapper):
        msa_file2 = [msa_file2]
    msa1 = MSA.from_file(msa_file1)
    metrics = []
    print_err(f"Running {msa_file1} against {len(msa_file2)} MSAs...")
    for msa2 in tqdm(msa_file2):
        if msa2 is not None:
            msa2 = MSA.from_file(msa2)
        metrics.append(
            build_evaluate_and_save_model(
                msa1=msa1,
                msa2=msa2,
                output_dir=output_dir,
                pdockq_t=pdockq_t,
                force_save=force_save,
                seed=seed,
                model_runner=model_runner,
            )
        )

    return metrics
    

def model_many_vs_many(msa_files1: Iterable[Union[str, TextIOWrapper]],
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
        metrics += model_one_vs_many(
                msa_file1=msa_file1,
                msa_file2=msa_files2,
                output_dir=output_dir,
                pdockq_t=pdockq_t,
                force_save=force_save,
                seed=seed,
                model_runner=model_runner,
            )
    return metrics