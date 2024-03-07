"""Command-line interface for sppid."""

__version__ = '0.0.1'

from typing import Any, Mapping, Optional, Tuple, Union

from argparse import ArgumentParser, FileType, Namespace
import sys

from carabiner import print_err
from carabiner.cliutils import clicommand, CLIOption, CLICommand, CLIApp

from .io import write_metrics
from .screening import build_evaluate_and_save_model, many_vs_many



@clicommand(message="Testing one PPI with the following parameters")
def _single(args: Namespace) -> None:

    metric = build_evaluate_and_save_model(
        msa_file1=args.msa1,
        msa_file2=args.msa2,
        max_recycles=args.recycles,
        output_dir=args.output,
        param_dir=args.params,
    )

    output_filename = os.path.join(output_dir, f"{metric.ID}_metrics.csv")
    print_err(f"Saving metrics as {output_filename}")
    write_metrics(metric, 
                  filename=output_filename)

    return None


@clicommand(message="Testing sets of PPIs with the following parameters")
def _many_vs_many(args: Namespace) -> None:

    metrics = many_vs_many(
        msa_files1=args.msa1,
        msa_files2=args.msa2,
        output_dir=args.output,
        max_recycles=args.recycles,
        param_dir=args.params,
    )

    output_filename = os.path.join(output_dir, "_all_metrics.csv")
    print_err(f"Saving metrics as {output_filename}")
    write_metrics(metrics, 
                  filename=output_filename)

    return None


def main() -> None:
    inputs = CLIOption('msa1', 
                       default=sys.stdin,
                       type=FileType('r'), 
                       nargs='?',
                       help='MSA file.')
    input2 = CLIOption('--msa2', '-2', 
                       type=FileType('r'), 
                       default=None,
                       help='Second MSA file.')
    inputs_list = CLIOption('msa1', 
                            default=sys.stdin,
                            type=FileType('r'), 
                            nargs='*',
                            help='MSA file(s).')
    inputs_list2 = CLIOption('--msa2', '-2', 
                        type=FileType('r'), 
                        default=None,
                        nargs='*',
                        help='Second MSA file(s). Default: if not provided, all pairwise from msa1.')
    output = CLIOption('--output', '-o', 
                       type=str,
                       required=True,
                       help='Output directory.')
    params = CLIOption('--params', '-p', 
                       type=str,
                       default=None,
                       help='Path to AlphaFold2 params file (.npz).')
    recycles = CLIOption('--recycles', '-x', 
                       type=int,
                       default=10,
                       help='Maximum number of recyles through the model.')

    single = CLICommand('single', 
                        description='Identify one protein-protein interaction.',
                        main=_single,
                        options=[inputs, input2, output, params, recycles])
    set_vs_set = CLICommand('many', 
                            description='Measure all interactions between two sets of proteins, or all pairs in one set of proteins.',
                            main=_many_vs_many,
                            options=[inputs_list, inputs_list2, output, params, recycles])

    app = CLIApp("sppid",
                 version=__version__,
                 description="Predicting protein-protein interactions using AlphaFold2.",
                 commands=[single, set_vs_set])

    app.run()
    return None


if __name__ == "__main__":
    main()