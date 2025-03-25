# Gene Ontology Term Prediction with GRPO

This module implements Generalized Reward-Penalty Optimization (GRPO) for predicting Gene Ontology (GO) terms for proteins based on their amino acid sequences. The model is trained to generate GO term predictions that match the ground truth annotations in the training dataset.

## Overview

The Gene Ontology (GO) is a concept hierarchy that describes the biological function of genes and gene products at different levels of abstraction. It is organized into three subontologies:

1. **Molecular Function (MFO)**: What a protein does at the molecular level
2. **Biological Process (BPO)**: Biological processes the protein participates in
3. **Cellular Component (CCO)**: Where in the cell the protein is located

This implementation focuses on predicting Molecular Function (MFO) terms for proteins based on their sequences.

## Required Files

The following files are required to run the training:

- `train_sequences.fasta`: Protein sequences in FASTA format
- `train_terms.tsv`: GO term annotations for proteins (UniProt ID, GO term ID, aspect)
- `go-basic.obo`: GO ontology structure

You can download these files using the provided script:

```bash
bash scripts/download_go_data.sh
```

## Running the Training

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the training script:

```bash
cd function
python functional_grpo.py
```

## Model Training

The training process:

1. Loads protein sequences, GO term annotations, and GO structure
2. Constructs prompts for each protein
3. Trains a model using GRPO to predict GO terms
4. Rewards predictions that match the ground truth GO terms

## Reward Function

The reward function uses:

- Precision, recall, and F1 score comparing predicted GO terms with ground truth
- Reward for properly formatted output using \boxed{} notation
- Small reward for sufficient thinking length

## Model Output Format

The model generates outputs in the following format:

```
<think>
[Detailed reasoning about protein function based on sequence analysis]
</think>
<answer>
Based on my analysis, this protein likely has the following GO terms:
- GO:0003674 (molecular_function): [reasoning]
- GO:0005515 (protein binding): [reasoning]
- GO:0046872 (metal ion binding): [reasoning]

\boxed{GO:0003674,GO:0005515,GO:0046872}
</answer>
```

The reward function extracts GO terms from the \boxed{} notation and compares them to the ground truth.

## Configuration

You can adjust the following parameters in the script:

- `MAX_EXAMPLES`: Maximum number of examples to process
- `NUM_EPOCHS`: Number of training epochs
- `GO_PREDICTION_REWARD`: Reward multiplier for correct predictions
- `FORMATTED_OUTPUT_REWARD`: Reward for properly formatted output
- `MAX_INPUT_LENGTH` and `MAX_OUTPUT_LENGTH`: Model sequence length limits 