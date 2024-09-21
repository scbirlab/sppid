# 🍐 yunta

![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/scbirlab/yunta/python-publish.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/yunta)
![PyPI](https://img.shields.io/pypi/v/yunta)

Predicting a pairwise protein-protein interactions and structures from multiple sequence alignments.

**`yunta`** provides several implementations of protein-protein interaction evaluation. In increasing computational cost:

- GPU-accelerated direct coupling analysis (DCA) (in Tensorflow and PyTorch)
- [RoseTTAFold](https://github.com/RosettaCommons/RoseTTAFold)-2track via the `rf2t-micro` package
- AlphaFold2 for protein-protein structure prediction

`yunta` has streamlined installation, a command-line interface, a Python API, and resilience to out-of-memory error. It takes as input unpaired multiple-sequence alignments in A3M format (as generated by tools like [`hhblits`](https://github.com/soedinglab/hh-suite)), and outputs a matrix of inter-residue contacts ()

## Installation

Obtaining and setting up `yunta` is easy.

```bash
$ pip install git+https://github.com/scbirlab/yunta
```

Using the embedded model requires using the [RoseTTAFold](https://github.com/RosettaCommons/RoseTTAFold)-2track weights. These are automatically downloaded, but by using `yunta` you agree that the trained weights for RoseTTAFold are made available for non-commercial use only under the terms of the [Rosetta-DL Software license](https://files.ipd.uw.edu/pub/RoseTTAFold/Rosetta-DL_LICENSE.txt).

AlphaFold2's pretrained parameters fall under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode), a copy of which is downloaded to `.weights/af2/` if AlphaFold2 models are used by `yunta`.

## Credit

`yunta` is a fork of [SpeedPPI](https://www.biorxiv.org/content/10.1101/2023.04.15.536993v1), which is itself inspired by [FoldDock](https://www.nature.com/articles/s41467-022-28865-w). This method used AlphaFold2 to evaluate 65,484 protein-protein interactions from the human proteome in [Towards a structurally resolved human protein interaction network](https://www.nature.com/articles/s41594-022-00910-8)]

## Usage 

You can always get more help by running

```bash
$ yunta --help
usage: yunta [-h] {dca-single,dca-many,rf2t-single,af2-single,af2-many} ...

Screening protein-protein interactions using DCA and AlphaFold2.

options:
  -h, --help            show this help message and exit

Sub-commands:
  {dca-single,dca-many,rf2t-single,af2-single,af2-many}
                        Use these commands to specify the tool you want to use.
    dca-single          Calculate DCA for one protein-protein interaction.
    dca-many            Calculate DCA between two sets of proteins, or all pairs in one set of proteins.
    rf2t-single         Calculate RF-2track contacts for between one protein and a series of others.
    af2-single          Model one protein-protein interaction.
    af2-many            Model all interactions between two sets of proteins, or all pairs in one set of proteins.
```

Once you have your MSAs file(s), you can run 1-vs-many with the `*-single` commands. For example:

```bash
$ yunta dca-single --help
usage: yunta dca-single [-h] [--msa2 [MSA2 ...]] [--list-file] [--output [OUTPUT]] [--plot PLOT] [--apc] [msa1]

positional arguments:
  msa1                  MSA file. Default: STDIN.

options:
  -h, --help            show this help message and exit
  --msa2 [MSA2 ...], -2 [MSA2 ...]
                        Second MSA file(s). Default: if not provided, all pairwise from msa1.
  --list-file, -l       Treat inputs as plain-text list of MSA files, rather than MSA filenames. Default: treat as MSA filenames.
  --output [OUTPUT], -o [OUTPUT]
                        Output filename. Default: STDOUT.
  --plot PLOT, -p PLOT  Directory for saving plots. Default: don't plot.
  --apc, -a             Whether to use APC correction in DCA. Default: don't apply correction.
```

If one MSA is provided, then homodimeric interactions are probed. For convenience, you can use the `--list-file` option to provide a single file containing a list of MSA files (one per line).

You can run many-vs-many with the `*-many` commands. For example:

```bash
$ yunta af2-many --help
usage: yunta af2-many [-h] [--msa2 [MSA2 ...]] [--list-file] --output OUTPUT [--params PARAMS] [--recycles RECYCLES] [--plot PLOT] [msa1 ...]

positional arguments:
  msa1                  MSA file(s). Default: "<_io.TextIOWrapper name='<stdin>' mode='r' encoding='utf-8'>".

options:
  -h, --help            show this help message and exit
  --msa2 [MSA2 ...], -2 [MSA2 ...]
                        Second MSA file(s). Default: if not provided, all pairwise from msa1.
  --list-file, -l       Treat inputs as plain-text list of MSA files, rather than MSA filenames. Default: treat as MSA filenames.
  --output OUTPUT, -o OUTPUT
                        Output directory. Required.
  --params PARAMS, -w PARAMS
                        Path to AlphaFold2 params file (.npz).
  --recycles RECYCLES, -x RECYCLES
                        Maximum number of recyles through the model. Default: "10".
  --plot PLOT, -p PLOT  Directory for saving plots. Default: don't plot.
```

## Python API

We provide an API for using MSAs in your own programs. 

```python
>>> from yunta.structs.msa import *
>>> msa = MSA.from_file("my-msa-file.a3m")
>>> msa.neff()
6
```

We also provide a reusable GPU-accelerated Tensorflow implementation of DCA (adapted from [Humpreys, _Science_, 2021](https://doi.org/10.1126/science.abm4805)).

```python
>>> from yunta.dca import calculate_dca
>>> from yunta.structs.msa import *
>>> paired_msa = PairedMSA.from_file("my-msa-file1.a3m", "my-msa-file2.a3m")
>>> calculate_dca(paired_msa, apc=True, gpu=False)
```

In case you prefer, you can also import a PyTorch implementation (which anecdotally is faster on both CPU and GPU).

```python
>>> from yunta.dca_torch import calculate_dca
>>> from yunta.structs.msa import *
>>> paired_msa = PairedMSA.from_file("my-msa-file1.a3m", "my-msa-file2.a3m")
>>> calculate_dca(paired_msa, apc=True, gpu=False)
```

(More documentation coming soon!)

## ... if you want to scale up

While the `*-many` commands can deal with processing multiple possible protein-protein interactions, if you want to screen more than a few and have access to a HPC cluster then using our [`nf-ggi` Nextflow pipeline](https://github.com/scbirlab/nf-ggi) pipeline will be more efficient. 

## Issues, problems, suggestions

Add to the [issue tracker](https://www.github.com/scbirlab/yunta/issues).

## Further help

Here are the help pages of the software used by this pipeline.

- [RoseTTAFold](https://github.com/RosettaCommons/RoseTTAFold)
- [AlphaFold2](https://github.com/google-deepmind/alphafold)
- [rf2t-micro](https://github.com/scbirlab/rf2t-micro)