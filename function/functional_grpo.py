import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainerCallback, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import re
import json
import os
import random
import wandb
from dotenv import load_dotenv
from datasets import Dataset
import time
from accelerate import PartialState
from huggingface_hub import login
from bitsandbytes.optim import PagedAdamW32bit
import torch.nn as nn
from pathlib import Path
import shutil
from datetime import datetime
import math
import csv
from Bio import SeqIO
from typing import Dict, Set, List, Tuple
import numpy as np

from openai import OpenAI

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

load_dotenv()

NUM_EPOCHS = 5
MAX_INPUT_LENGTH = 6000
MAX_OUTPUT_LENGTH = 4096
THINK_LENGTH = 3000
DEVICE = "cuda"
RUN_NAME = "cafa_functional_go_prediction" # FILL IN RUN NAME FOR wandb
CHECKPOINT_PATH = "none" 

GO_PREDICTION_REWARD = 1.0
FORMATTED_OUTPUT_REWARD = 0.2

# Configuration settings
TRAIN_SEQUENCES_PATH = "function/cafa/Train/train_sequences.fasta" 
TRAIN_TERMS_PATH = "function/cafa/Train/train_terms.tsv"
GO_STRUCTURE_PATH = "function/cafa/Train/go-basic.obo"
TERM_WEIGHTS_PATH = "function/cafa/IA.txt"  # Updated path to IA.txt
MAX_EXAMPLES = 1000  # Limit number of examples to process

# Evidence codes for experimental validation
EXPERIMENTAL_EVIDENCE_CODES = {'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP'}

# Print model size information
def get_model_size_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get actual memory usage
    memory_params = sum(p.nelement() * p.element_size() for p in model.parameters())
    memory_buffers = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_memory = memory_params + memory_buffers  # in bytes
    
    # Convert to more readable formats
    def bytes_to_mb(bytes_val): return bytes_val / (1024 * 1024)
    def bytes_to_gb(bytes_val): return bytes_val / (1024 * 1024 * 1024)
    
    print(f"\nModel Size Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Actual model size in memory: {bytes_to_gb(total_memory):.2f} GB")
    
    # If using CUDA, also show GPU memory usage
    if next(model.parameters()).is_cuda:
        print("\nGPU Memory Usage:")
        print(f"Allocated: {bytes_to_gb(torch.cuda.memory_allocated()):.2f} GB")
        print(f"Cached: {bytes_to_gb(torch.cuda.memory_reserved()):.2f} GB")
        
        # Show per-tensor memory usage (top 5 largest tensors)
        print("\nLargest Model Tensors:")
        tensor_sizes = [(name, p.nelement() * p.element_size()) 
                       for name, p in model.named_parameters()]
        tensor_sizes.sort(key=lambda x: x[1], reverse=True)
        for name, size in tensor_sizes[:5]:
            print(f"{name}: {bytes_to_mb(size):.2f} MB")


def load_go_structure(go_structure_path):
    """Load GO structure and extract term information"""
    try:
        go_terms = {}
        current_term = None
        
        with open(go_structure_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '[Term]':
                    current_term = {}
                elif line.startswith('id:') and current_term is not None:
                    current_term['id'] = line.split('id:')[1].strip()
                elif line.startswith('name:') and current_term is not None:
                    current_term['name'] = line.split('name:')[1].strip()
                elif line.startswith('namespace:') and current_term is not None:
                    current_term['namespace'] = line.split('namespace:')[1].strip()
                elif line.startswith('def:') and current_term is not None:
                    current_term['def'] = line.split('def:')[1].strip()
                elif line.startswith('is_a:') and current_term is not None:
                    if 'is_a' not in current_term:
                        current_term['is_a'] = []
                    current_term['is_a'].append(line.split('is_a:')[1].strip().split('!')[0].strip())
                elif line == '' and current_term is not None and 'id' in current_term:
                    go_terms[current_term['id']] = current_term
                    current_term = None
        
        # print(f"Loaded {len(go_terms)} GO terms")
        return go_terms
    except Exception as e:
        print(f"Error loading GO structure: {e}")
        return {}

def load_protein_annotations(train_terms_path: str) -> Dict[str, Dict[str, Dict[str, Set[str]]]]:
    """
    Load protein GO term annotations from tsv file, separating experimental and predicted annotations by aspect.
    
    Args:
        train_terms_path (str): Path to the TSV file containing protein annotations
        
    Returns:
        Dict with structure:
        {
            'MFO/BPO/CCO': {
                'experimental': {protein_id: set(terms)},
                'predicted': {protein_id: set(terms)}
            }
        }
    """
    try:
        protein_annotations = {
            'MFO': {'experimental': {}, 'predicted': {}},
            'BPO': {'experimental': {}, 'predicted': {}},
            'CCO': {'experimental': {}, 'predicted': {}}
        }
        
        with open(train_terms_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)  # Skip header row
            
            # Verify expected columns exist
            expected_cols = ['protein_id', 'term_id', 'aspect', 'evidence']
            if len(header) < len(expected_cols):
                raise ValueError(f"Missing columns in annotation file. Expected: {expected_cols}")
            
            for row in reader:
                if len(row) < 4:
                    print(f"Warning: Skipping malformed row: {row}")
                    continue
                    
                protein_id, term_id, aspect, evidence = row[:4]
                protein_id = protein_id.strip()
                term_id = term_id.strip()
                aspect = aspect.strip()
                evidence = evidence.strip()
                
                # Skip if aspect is not one of the main three
                if aspect not in protein_annotations:
                    continue
                
                # Determine if evidence is experimental
                is_experimental = evidence in EXPERIMENTAL_EVIDENCE_CODES
                annotation_type = 'experimental' if is_experimental else 'predicted'
                
                # Initialize sets if needed
                if protein_id not in protein_annotations[aspect][annotation_type]:
                    protein_annotations[aspect][annotation_type][protein_id] = set()
                
                # Add the term
                protein_annotations[aspect][annotation_type][protein_id].add(term_id)
        
        # Print statistics
        for aspect in protein_annotations:
            exp_count = len(protein_annotations[aspect]['experimental'])
            pred_count = len(protein_annotations[aspect]['predicted'])
            print(f"{aspect} annotations:")
            print(f"  Experimental: {exp_count} proteins")
            print(f"  Predicted: {pred_count} proteins")
        
        return protein_annotations
        
    except Exception as e:
        print(f"Error loading protein annotations: {e}")
        return {
            'MFO': {'experimental': {}, 'predicted': {}},
            'BPO': {'experimental': {}, 'predicted': {}},
            'CCO': {'experimental': {}, 'predicted': {}}
        }

def load_protein_sequences(train_sequences_path):
    """Load protein sequences from FASTA file"""
    try:
        protein_sequences = {}
        for record in SeqIO.parse(train_sequences_path, "fasta"):
            # Extract UniProt ID from the header
            header = record.description  # Use description instead of id to get full header
            # Try different patterns to extract UniProt ID
            if 'sp|' in header:
                uniprot_id = header.split('sp|')[1].split('|')[0]
            elif 'tr|' in header:
                uniprot_id = header.split('tr|')[1].split('|')[0]
            else:
                uniprot_id = header.split()[0]  # Fall back to first word
                
            protein_sequences[uniprot_id] = str(record.seq)
        
        print(f"Loaded sequences for {len(protein_sequences)} proteins")
        return protein_sequences
    except Exception as e:
        print(f"Error loading protein sequences: {e}")
        print(f"Error details: header={header if 'header' in locals() else 'N/A'}")
        return {}

def construct_prompt(
    protein_id: str,
    sequence: str,
    protein_annotations: Dict,
    go_terms: Dict,
    target_aspect: str
) -> str:
    """
    Construct prompt for GO term prediction for a specific aspect.
    
    Args:
        protein_id: Protein identifier
        sequence: Amino acid sequence
        protein_annotations: Dictionary containing protein annotations
        go_terms: Dictionary containing GO term information
        target_aspect: Target aspect to predict (MFO, BPO, or CCO)
    """
    # Get protein description
    protein_description = f"Protein ID: {protein_id}"
    
    # Format known annotations for each aspect
    def format_terms(terms: Set[str], aspect_name: str) -> str:
        if not terms:
            return f"  No experimentally validated {aspect_name} terms known"
        formatted = []
        for term_id in sorted(terms)[:3]:  # Show up to 3 terms, sorted for consistency
            term_info = go_terms.get(term_id, {})
            name = term_info.get('name', 'Unknown term')
            definition = term_info.get('def', '').split('"')[1] if 'def' in term_info else ''
            formatted.append(f"  - {term_id}: {name}")
            if definition:
                formatted.append(f"    Definition: {definition}")
        return "\n".join(formatted)
    
    # Get experimental annotations for non-target aspects
    other_aspects = [asp for asp in ['MFO', 'BPO', 'CCO'] if asp != target_aspect]
    aspect_annotations = {}
    for aspect in other_aspects:
        terms = protein_annotations.get(aspect, {}).get('experimental', {}).get(protein_id, set())
        aspect_annotations[aspect] = format_terms(terms, aspect)
    
    # Map aspect codes to full names
    aspect_names = {
        'MFO': 'Molecular Function',
        'BPO': 'Biological Process',
        'CCO': 'Cellular Component'
    }
    
    go_prompt = f"""You are an expert in protein function prediction using the Gene Ontology (GO) framework. Your task is to predict {aspect_names[target_aspect]} terms for this protein based on its sequence and known annotations in other aspects.

Target Aspect: {target_aspect}

PROTEIN INFORMATION:
{protein_description}

PROTEIN SEQUENCE:
{sequence}

KNOWN FUNCTIONAL ANNOTATIONS:
"""

    # Add annotations for non-target aspects
    for aspect in other_aspects:
        go_prompt += f"\n{aspect_names[aspect]} ({aspect}) - describes {aspect_names[aspect].lower()}:\n"
        go_prompt += f"{aspect_annotations[aspect]}\n"

    go_prompt += f"""
Based on:
1. The protein's amino acid sequence
2. Known annotations in other aspects
3. Your knowledge of protein domains, motifs, and cellular organization

Predict the most likely {aspect_names[target_aspect]} GO terms for this protein. Consider:
- A protein can have multiple functions and participate in multiple processes
- GO terms form a hierarchy (more specific terms are preferred over general ones)
- Focus on specific, detailed predictions rather than just high-level terms
- Only predict terms that you are confident are supported by the sequence or known annotations

Your final answer should include only the predicted GO term IDs in a comma-separated list enclosed in \\boxed{{}} notation.
Example: \\boxed{{GO:0003674,GO:0005515,GO:0046872}}

First, think through your reasoning process considering:
1. Sequence features (length, composition, motifs)
2. Known annotations and their implications for {aspect_names[target_aspect]}
3. Cellular context and biological roles
4. Potential molecular mechanisms

Then provide your predictions."""

    whole_prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that helps users with protein function prediction. You first think about the reasoning process and then provide the answer. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively. Your thinking should be at least 3000 tokens. |eot_id|><|start_header_id|>user<|end_header_id|>
{go_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    return whole_prompt

def create_dataset(
    protein_sequences: Dict[str, str],
    protein_annotations: Dict[str, Dict[str, Dict[str, Set[str]]]],
    go_terms: Dict[str, Dict],
    max_examples: int = 1000
) -> List[Dict]:
    """
    Create dataset for training, handling each aspect separately.
    
    Args:
        protein_sequences: Dictionary mapping protein IDs to sequences
        protein_annotations: Dictionary containing protein annotations by aspect
        go_terms: Dictionary containing GO term information
        max_examples: Maximum number of examples to include
    """
    dataset_records = []
    
    # Get proteins with sequences
    proteins_with_sequences = set(protein_sequences.keys())
    print(f"Found {len(proteins_with_sequences)} proteins with sequences")
    
    # Create examples for each aspect
    for aspect in ['MFO', 'BPO', 'CCO']:
        # Get proteins with experimental annotations for this aspect
        proteins_with_exp = set(protein_annotations[aspect]['experimental'].keys())
        proteins_without_exp = proteins_with_sequences - proteins_with_exp
        
        print(f"\n{aspect}:")
        print(f"  Proteins with experimental annotations: {len(proteins_with_exp)}")
        print(f"  Proteins without experimental annotations: {len(proteins_without_exp)}")
        
        # Create records for proteins without experimental annotations in this aspect
        for protein_id in list(proteins_without_exp)[:max_examples // 3]:  # Limit per aspect
            record = {
                "prompt": construct_prompt(
                    protein_id,
                    protein_sequences[protein_id],
                    protein_annotations,
                    go_terms,
                    target_aspect=aspect
                ),
                "protein_id": protein_id,
                "aspect": aspect,
                "sequence": protein_sequences[protein_id],
                "ground_truth_terms": list(protein_annotations[aspect]['experimental'].get(protein_id, set()))
            }
            dataset_records.append(record)
    
    print(f"\nCreated {len(dataset_records)} total dataset records")
    return dataset_records

def extract_go_terms_from_response(response):
    """Extract GO terms from model response"""
    try:
        # Look for terms in \boxed{...}
        boxed_match = re.search(r'\\boxed{([^}]+)}', response)
        if boxed_match:
            terms_str = boxed_match.group(1).strip()
            terms = [term.strip() for term in terms_str.split(',')]
            # Validate terms format (GO:XXXXXXX)
            terms = [term for term in terms if re.match(r'^GO:\d{7}$', term)]
            return terms
        
        # If boxed format not found, try looking in the answer tags
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            # Find all GO:XXXXXXX patterns
            terms = re.findall(r'GO:\d{7}', answer)
            return terms
        
        print("Warning: No valid GO terms found in the expected format")
        return []
        
    except Exception as e:
        print(f"Error extracting GO terms: {e}")
        return []

def calculate_weighted_fmeasure(
    predicted_terms: Set[str],
    true_terms: Set[str],
    term_weights: Dict[str, float],
    aspect: str
) -> Tuple[float, float, float]:
    """
    Calculate the weighted F-measure for GO term predictions using information accretion weights.
    
    Args:
        predicted_terms: Set of predicted GO terms
        true_terms: Set of true (ground truth) GO terms
        term_weights: Dictionary mapping GO terms to their information accretion weights
        aspect: The GO aspect being evaluated (MFO, BPO, or CCO)
        
    Returns:
        Tuple of (weighted F1, weighted precision, weighted recall)
    """
    # Convert inputs to sets if they aren't already
    predicted_terms = set(predicted_terms)
    true_terms = set(true_terms)
    
    # Calculate weighted true positives, false positives, and false negatives
    weighted_tp = sum(term_weights.get(term, 0.0) for term in (predicted_terms & true_terms))
    weighted_fp = sum(term_weights.get(term, 0.0) for term in (predicted_terms - true_terms))
    weighted_fn = sum(term_weights.get(term, 0.0) for term in (true_terms - predicted_terms))
    
    # Calculate weighted precision and recall
    weighted_precision = weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp) > 0 else 0.0
    weighted_recall = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) > 0 else 0.0
    
    # Calculate weighted F1 score
    if weighted_precision + weighted_recall > 0:
        weighted_f1 = (2 * weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
    else:
        weighted_f1 = 0.0
        
    return weighted_f1, weighted_precision, weighted_recall

def go_prediction_reward_func(prompts, completions, **kwargs):
    """Reward function for GO term prediction using weighted F-measure"""
    rewards = []
    
    # Load term weights if not already loaded
    if not hasattr(go_prediction_reward_func, 'term_weights'):
        print("\nLoading term weights...")
        start_time = time.time()
        go_prediction_reward_func.term_weights = load_term_weights(TERM_WEIGHTS_PATH)
        print(f"Term weights loaded in {time.time() - start_time:.2f} seconds")
    
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        try:
            reward = 0.0
            
            # Print completion for analysis
            print(f"\nCOMPLETION {i}")
            print(completion)
            print('-'*100)
            
            # Extract thinking for logging
            think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
            thinking = think_match.group(1).strip() if think_match else "No thinking found"
            thinking_length = len(thinking.split())
            
            # Extract predicted GO terms
            predicted_terms = set(extract_go_terms_from_response(completion))
            
            # Extract aspect and ground truth terms from the prompt
            aspect_match = re.search(r'Target Aspect: (MFO|BPO|CCO)', prompt)
            aspect = aspect_match.group(1) if aspect_match else None
            
            ground_truth_match = re.search(r'ground_truth_terms: \[(.*?)\]', prompt)
            true_terms = set()
            if ground_truth_match:
                terms_str = ground_truth_match.group(1)
                true_terms = {term.strip(' "\'') for term in terms_str.split(',')}
            
            if aspect and true_terms:
                # Calculate weighted F-measure
                weighted_f1, weighted_precision, weighted_recall = calculate_weighted_fmeasure(
                    predicted_terms,
                    true_terms,
                    go_prediction_reward_func.term_weights,
                    aspect
                )
                
                # Base reward on weighted F1 score
                reward += GO_PREDICTION_REWARD * weighted_f1
                
                # Additional reward for properly formatted output
                if re.search(r'\\boxed{([^}]+)}', completion):
                    reward += FORMATTED_OUTPUT_REWARD
                
                # Print detailed comparison for every completion
                print(f"\nCompletion {i} Analysis:")
                print(f"Aspect: {aspect}")
                print(f"Predicted Terms: {predicted_terms}")
                print(f"True Terms: {true_terms}")
                print(f"Weighted F1: {weighted_f1:.4f}")
                print(f"Total Reward: {reward:.4f}")
                print("\nThinking excerpt (first 200 chars):")
                print(thinking[:200] + "...")
                print("\nCompletion excerpt (first 200 chars):")
                print(completion[:200] + "...")
                print('-'*100)
                
                # Log detailed metrics
                wandb.log({
                    f"reward/completion_{i}/total_reward": reward,
                    f"reward/completion_{i}/weighted_precision": weighted_precision,
                    f"reward/completion_{i}/weighted_recall": weighted_recall,
                    f"reward/completion_{i}/weighted_f1": weighted_f1,
                    f"reward/completion_{i}/predicted_terms_count": len(predicted_terms),
                    f"reward/completion_{i}/true_terms_count": len(true_terms),
                    f"reward/completion_{i}/thinking_length": thinking_length,
                    f"reward/completion_{i}/aspect": aspect
                })
            
            rewards.append(reward)
                        
        except Exception as e:
            print(f"Error calculating rewards: {e}")
            rewards.append(0.0)
    
    return rewards

class WandBLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        proc_state = PartialState()
        if proc_state.is_main_process and logs:  #Only log on main process
            # Log all metrics from the trainer
            wandb.log(logs, step=state.global_step)
            
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        proc_state = PartialState()
        if proc_state.is_main_process and metrics:  # Only log on main process
            # Log evaluation metrics
            wandb.log({"eval/" + k: v for k, v in metrics.items()}, step=state.global_step) 

class CheckpointCallback(TrainerCallback):
    def __init__(self, checkpoint_dir="checkpoints", checkpoint_freq=100, max_checkpoints=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_freq = checkpoint_freq
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def on_step_end(self, args, state, control, **kwargs):
        """Save checkpoint every checkpoint_freq steps"""
        if state.global_step % self.checkpoint_freq == 0:
            self._save_checkpoint(args, state)
            
    def _save_checkpoint(self, args, state):
        """Save LoRA checkpoint and maintain max number of checkpoints"""
        proc_state = PartialState()
        if not proc_state.is_main_process:
            return
            
        # Create checkpoint name with timestamp and step
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_name = f"checkpoint-{timestamp}-step{state.global_step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        try:
            # Save LoRA weights and config
            state.model.save_pretrained(checkpoint_path)  # This saves LoRA weights
            
            # Save additional training state
            training_state = {
                "global_step": state.global_step,
                "epoch": state.epoch,
                "best_metric": state.best_metric,
                "training_args": args.to_dict(),
            }
            torch.save(training_state, checkpoint_path / "trainer_state.pt")
            
            # Save tokenizer
            tokenizer.save_pretrained(checkpoint_path)
            
            # Maintain only max_checkpoints number of checkpoints
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint-*"))
            if len(checkpoints) > self.max_checkpoints:
                for checkpoint in checkpoints[:-self.max_checkpoints]:
                    shutil.rmtree(checkpoint)
                    
            print(f"Saved LoRA checkpoint: {checkpoint_path}")
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

def load_from_checkpoint(checkpoint_path, model, trainer):
    """Load LoRA weights and training state from checkpoint"""
    try:
        checkpoint_path = Path(checkpoint_path)
        
        # Load LoRA weights
        model.load_adapter(checkpoint_path, "default")  # Load LoRA weights
        
        # Load training state
        training_state = torch.load(checkpoint_path / "trainer_state.pt")
        trainer.state.global_step = training_state["global_step"]
        trainer.state.epoch = training_state["epoch"]
        trainer.state.best_metric = training_state["best_metric"]
        
        print(f"Loaded LoRA checkpoint from {checkpoint_path}")
        return True
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

def load_term_weights(weights_file: str) -> Dict[str, float]:
    """
    Load information accretion (IA) weights for GO terms.
    
    Args:
        weights_file (str): Path to the IA.txt file containing term weights
        
    Returns:
        Dict[str, float]: Dictionary mapping GO terms to their information accretion weights
    """
    term_weights = {}
    try:
        with open(weights_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        term, weight = line.strip().split('\t')
                        # Convert weight to float, handle potential scientific notation
                        weight = float(weight)
                        term_weights[term] = weight
                    except ValueError as ve:
                        print(f"Warning: Could not parse weight from line: {line.strip()}")
                        continue
                    except Exception as e:
                        print(f"Warning: Error processing line: {line.strip()}, Error: {e}")
                        continue
                        
        print(f"Loaded {len(term_weights)} term weights")
        # Print some statistics about the weights
        # if term_weights:
        #     weights = list(term_weights.values())
        #     print(f"Weight statistics:")
        #     print(f"  Min weight: {min(weights):.6f}")
        #     print(f"  Max weight: {max(weights):.6f}")
        #     print(f"  Mean weight: {sum(weights)/len(weights):.6f}")
        #     print(f"  Zero weights: {sum(1 for w in weights if w == 0.0)}")
            
        return term_weights
    except Exception as e:
        print(f"Error loading term weights: {e}")
        return {}

# Data loading section
data_load_start = time.time()

# Check if necessary files exist, if not, inform user they need to be downloaded
missing_files = []
if not os.path.exists(TRAIN_SEQUENCES_PATH):
    missing_files.append(TRAIN_SEQUENCES_PATH)
if not os.path.exists(TRAIN_TERMS_PATH):
    missing_files.append(TRAIN_TERMS_PATH)
if not os.path.exists(GO_STRUCTURE_PATH):
    missing_files.append(GO_STRUCTURE_PATH)

if missing_files:
    print(f"The following required files are missing: {', '.join(missing_files)}")
    print("Please download these files before running this script.")
    # For this example, we'll create a small dummy dataset
    valid_data_list = []
    for i in range(10):
        valid_data_list.append({
            "prompt": f"Dummy prompt {i}",
            "protein_id": f"P{i:05d}",
            "sequence": "MSFTAPVAEDSDF",
            "ground_truth_terms": [f"GO:{i:07d}"]
        })
else:
    # Load data
    go_terms = load_go_structure(GO_STRUCTURE_PATH)
    protein_annotations = load_protein_annotations(TRAIN_TERMS_PATH)
    protein_sequences = load_protein_sequences(TRAIN_SEQUENCES_PATH)
    
    # Create dataset
    valid_data_list = create_dataset(protein_sequences, protein_annotations, go_terms, max_examples=MAX_EXAMPLES)

# Create dataset from records
train_dataset = Dataset.from_list(valid_data_list)
print(f"Dataset size: {len(train_dataset)}")
print(f"Data loading completed in {time.time() - data_load_start:.2f} seconds")

# Print an example dataset entry
if len(train_dataset) > 0:
    print("\nExample Dataset Entry:")
    example_entry = train_dataset[0]
    print(f"Protein ID: {example_entry['protein_id']}")
    print(f"Sequence length: {len(example_entry['sequence'])}")
    print(f"Number of ground truth terms: {len(example_entry['ground_truth_terms'])}")
    print(f"Ground truth terms: {example_entry['ground_truth_terms']}")
    print("\nPrompt excerpt (first 500 chars):")
    print(example_entry['prompt'][:500] + "...")

# Calculate and print dataset statistics for prompt lengths
prompt_lengths = [len(example['prompt']) for example in valid_data_list]
print("\nPrompt Length Statistics:")
print(f"Mean length: {sum(prompt_lengths) / len(prompt_lengths):.2f}")
print(f"Median length: {sorted(prompt_lengths)[len(prompt_lengths)//2]}")
print(f"Max length: {max(prompt_lengths)}")
print(f"Min length: {min(prompt_lengths)}")


# Initialize wandb only on main process
proc_state = PartialState()
if proc_state.is_main_process:
    wandb_start = time.time()
    try:
        wandb.login(key=os.getenv('WANDB_API_KEY'))
        wandb.init(
            project="function-rl",
            name=RUN_NAME,
            config={
                "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct",
                "num_epochs": NUM_EPOCHS,
                "batch_size": 2,
                "learning_rate": 1e-3,
                "num_generations": 4,
                "continued_from_checkpoint": CHECKPOINT_PATH,
            }
        )
    except Exception as e:
        print(f"Error initializing wandb: {e}")
        # Login to Hugging Face before any model loading
    try:
        huggingface_token = os.getenv('HUGGINGFACE_API_KEY')
        if not huggingface_token:
            raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
        login(token=huggingface_token)
        print("Successfully logged into Hugging Face")
    except Exception as e:
        print(f"Error logging into Hugging Face: {e}")
        raise  # Add this to stop execution if login fails

# Initialize Unsloth's FastLanguageModel with GRPO patch
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

# Model initialization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/meta-Llama-3.1-8B-Instruct",
    max_seq_length=MAX_INPUT_LENGTH,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=32,  # Adjust based on your needs
    gpu_memory_utilization=0.6,
)

# Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Modify training arguments for Unsloth compatibility
training_args = GRPOConfig(
    use_vllm=False,
    learning_rate=2e-4,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=2,
    max_prompt_length=MAX_INPUT_LENGTH,
    max_completion_length=MAX_OUTPUT_LENGTH,
    num_train_epochs=NUM_EPOCHS,
    max_grad_norm=0.1,
    output_dir=f"./{RUN_NAME}",
)

# Initialize GRPO trainer with Unsloth configuration
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[go_prediction_reward_func],  # Use the GO prediction reward function
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[WandBLoggingCallback(),CheckpointCallback(
        checkpoint_dir=f"./{RUN_NAME}/checkpoints",
        checkpoint_freq=8, 
        max_checkpoints=5     # Keep last 5 checkpoints
    )]
)

# Train the model
trainer.train()

# Save the model - Unsloth style
model.save_pretrained(f"./{RUN_NAME}/final_model")

if proc_state.is_main_process:
    wandb.finish()

