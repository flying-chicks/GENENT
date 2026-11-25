# GENET

## Generation-Based Statistical Testing of Explanatory Subgraph Effects on GNN Predictions

**A generation-based hypothesis testing framework for evaluating the statistical significance of GNN explanatory subgraphs.**

<!-- [Paper](TODO_PAPER_URL) ·  -->
[Overview](#overview) · [Getting Started](#getting-started)
 <!-- · [Citation](#citation) -->

<!-- > **Repository setup:** Replace the `TODO` placeholders with the actual paper links, dependencies, execution commands, and citation details before publishing. -->

## Overview

Graph neural network (GNN) explainers identify subgraphs that help explain model predictions. However, identifying an explanatory subgraph alone does not establish that its effect is statistically significant. Without formal hypothesis testing, the statistical evidence supporting these explanations remains limited.

**GENET** addresses this gap by providing a hypothesis testing framework for subgraphs produced by existing GNN explainers. It combines **discrete graph diffusion**, **conditional average treatment effect (CATE) estimation**, and a **paired sign-flip test** to assess the statistical significance of explanatory subgraph effects on GNN predictions.

GENET complements existing explainers rather than replacing them: an explainer proposes a subgraph, and GENET evaluates the statistical evidence for its effect.

## Key Features

- **Statistical evaluation of explanations:** Assess whether explanatory subgraphs have statistically significant effects on GNN predictions.
- **Generation-based graph sampling:** Learn the original graph data distribution using a discrete graph diffusion model.
- **Paired effect estimation:** Estimate CATE using sampled control–treatment graph pairs.
- **Hypothesis testing:** Evaluate statistical significance through a paired sign-flip test.
- **Theoretical and empirical support:** The paper provides theoretical analysis and experimental evaluation of the framework.

## Method

### Pipeline

```text
Original Graph Dataset
         │
         ▼
Train a Discrete Graph Diffusion Model
         │
         ▼
Sample Control–Treatment Graph Pairs ◄── Explanatory Subgraph
         │
         ▼
Evaluate Predictions with the Target GNN
         │
         ▼
Estimate Conditional Average Treatment Effect
         │
         ▼
Perform a Paired Sign-Flip Test
         │
         ▼
Report Effect Estimate and Statistical Significance
```

### 1. Learn the Graph Distribution

GENET trains a discrete graph diffusion model to capture the original data distribution. The pretrained model supports the subsequent construction of control–treatment graph pairs.

### 2. Sample Control–Treatment Pairs

Given an explanatory subgraph, GENET samples paired graphs under the control and treatment conditions defined by the framework.


### 3. Estimate the Subgraph Effect & Test Statistical Significance
 GENET evaluates the target GNN on the sampled graph pairs and uses the paired prediction differences to estimate the conditional average treatment effect of the explanatory subgraph.
GENET applies a paired sign-flip test to assess the statistical significance of the estimated effect.


> Statistical validity depends on the assumptions and sampling procedure described in the paper. A significant result supports the tested effect under this setup; it does not, by itself, establish that an explanation is unique or complete.

## Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/flying-chicks/GENENT
cd GENET
```

**TODO:** Add the supported Python version, required packages, and installation commands.

### Data and Model Preparation

Prepare the following inputs:

- **Graph dataset:** MUTAG and BA2.
- **Target GNN:** src/gnn_models/ba2motif_gnn.py, src/gnn_models/mutag_gnn.py.
- **Explanatory subgraphs:** Subgraphs identified by an existing GNN explainer.

### Platforms
- Download anaconda/miniconda if needed
- Create a python environment:
```bash
conda create -c conda-forge -n GENET python=3.9
conda activate GENET
```
- Install the nvcc drivers for your cuda version and pytorch:
```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda
pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```


### Running GENET

The evaluation workflow consists of:

1. Train or load a discrete graph diffusion model.
```bash
cd GENET
python src/main.py dataset=MUTAG
```

2. Sample control–treatment graph pairs.
```bash
cd GENET
python src/generate.py n_pairs=1000
```
3. Estimate the subgraph effects on GNN predictions and run the paired sign-flip test and report the results.
```bash
cd GENET
python src/flip_test.py
```


<!-- **TODO:** Add executable commands and configuration examples for each step. -->

## Results and Reproducibility
Unfortunately, my Google Drive account has been deleted, and I no longer have access to the checkpoints. If you have a checkpoint saved locally, I would greatly appreciate it if you could contact me at yuguanyuan@swufe.edu.cn or open a GitHub issue.



<!-- The paper evaluates GENET through theoretical analysis and empirical experiments. -->

<!-- **TODO:** Add benchmark results, pretrained checkpoints, and reproduction instructions. -->

<!-- For reproducible evaluation, document:

- Dataset splits and preprocessing settings.
- Target GNN and explainer configurations.
- Diffusion model training and sampling settings.
- Number of sampled graph pairs.
- Sign-flip test configuration and significance threshold.
- Random seeds and software versions. -->

<!-- ## Citation

If you use GENET in your research, please cite:

```bibtex
@article{genet,
  title   = {GENET: Generation-Based Statistical Testing of Explanatory Subgraph Effects on GNN Predictions},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

**TODO:** Replace this placeholder with the official BibTeX entry. -->

<!-- ## License

**TODO:** Specify the project license and add a `LICENSE` file. -->

## Contact

For questions or bug reports, please open a GitHub issue.

- yuguanyuan@swufe.edu.cn
