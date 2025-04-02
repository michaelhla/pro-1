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
MAX_EXAMPLES = 1000  # Limit number of examples to process

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
        
        print(f"Loaded {len(go_terms)} GO terms")
        return go_terms
    except Exception as e:
        print(f"Error loading GO structure: {e}")
        return {}

def load_protein_annotations(train_terms_path):
    """Load protein GO term annotations from tsv file"""
    try:
        protein_annotations = {}
        with open(train_terms_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # Skip header row
            for row in reader:
                if len(row) >= 3:
                    protein_id = row[0].strip()
                    term_id = row[1].strip()
                    aspect = row[2].strip()
                    if protein_id not in protein_annotations:
                        protein_annotations[protein_id] = {'MFO': [], 'BPO': [], 'CCO': []}
                    protein_annotations[protein_id][aspect].append(term_id)
        
        print(f"Loaded annotations for {len(protein_annotations)} proteins")
        return protein_annotations
    except Exception as e:
        print(f"Error loading protein annotations: {e}")
        print(f"Error details: row={row if 'row' in locals() else 'N/A'}")
        return {}

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

def construct_prompt(protein_id, sequence, mfo_terms, bpo_terms, cco_terms, go_terms):
    """Construct prompt for GO term prediction"""
    
    # Get protein description from FASTA header if available
    protein_description = f"Protein ID: {protein_id}"
    
    # Format the known GO terms for each aspect
    mfo_text = "\n".join([f"  - {term_id}: {go_terms.get(term_id, {}).get('name', 'Unknown term')}" for term_id in mfo_terms[:3]]) if mfo_terms else "  None"
    bpo_text = "\n".join([f"  - {term_id}: {go_terms.get(term_id, {}).get('name', 'Unknown term')}" for term_id in bpo_terms[:3]]) if bpo_terms else "  None"
    cco_text = "\n".join([f"  - {term_id}: {go_terms.get(term_id, {}).get('name', 'Unknown term')}" for term_id in cco_terms[:3]]) if cco_terms else "  None"
    
    # Construct the prompt
    go_prompt = f"""You have been studying protein functions for decades. Given a protein sequence, predict its Gene Ontology (GO) terms.

PROTEIN INFORMATION:
{protein_description}

PROTEIN SEQUENCE:
{sequence}

KNOWN GO TERMS (PARTIAL LIST):
Molecular Function (MFO):
{mfo_text}

Biological Process (BPO):
{bpo_text}

Cellular Component (CCO):
{cco_text}

Based on the protein sequence and any known GO terms, predict additional GO terms for this protein. Focus on the most likely Molecular Function (MFO) terms. For each prediction, provide:
1. The GO term ID
2. The GO term name
3. Your reasoning based on sequence analysis, known protein domains, and similar proteins

Your final answer should include only the GO term IDs in a comma-separated list enclosed in \\boxed{{}} notation.
Example: \\boxed{{GO:0003674,GO:0005515,GO:0046872}}
"""

    whole_prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that helps users with protein function prediction. You first think about the reasoning process and then provide the answer. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively. Your thinking should be at least 3000 tokens. |eot_id|><|start_header_id|>user<|end_header_id|>
{go_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    return whole_prompt

def create_dataset(protein_sequences, protein_annotations, go_terms, max_examples=1000):
    """Create dataset for training"""
    dataset_records = []
    
    # Get proteins with both sequence and annotations
    common_proteins = set(protein_sequences.keys()) & set(protein_annotations.keys())
    print(f"Found {len(common_proteins)} proteins with both sequence and annotations")
    
    # Limit to max_examples
    proteins_to_use = list(common_proteins)[:max_examples]
    
    for protein_id in proteins_to_use:
        sequence = protein_sequences[protein_id]
        annotations = protein_annotations[protein_id]
        
        # Sample some terms for the prompt and keep others as ground truth
        mfo_terms = annotations.get('MFO', [])
        bpo_terms = annotations.get('BPO', [])
        cco_terms = annotations.get('CCO', [])
        
        # All MFO terms are the ground truth labels we want to predict
        ground_truth_terms = mfo_terms
        
        # Skip proteins with no MFO terms
        if not ground_truth_terms:
            continue
        
        # Create the dataset record
        record = {
            "prompt": construct_prompt(protein_id, sequence, [], bpo_terms, cco_terms, go_terms),
            "protein_id": protein_id,
            "sequence": sequence,
            "ground_truth_terms": ground_truth_terms
        }
        
        dataset_records.append(record)
    
    print(f"Created {len(dataset_records)} dataset records")
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

def go_prediction_reward_func(prompts, completions, protein_ids, ground_truth_terms, **kwargs):
    """Reward function for GO term prediction"""
    rewards = []
    
    for i, (prompt, completion, protein_id, terms) in enumerate(zip(prompts, completions, protein_ids, ground_truth_terms)):
        try:
            reward = 0.0
            print(f"COMPLETION {i}")
            
            # Extract thinking for logging
            think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
            thinking = think_match.group(1).strip() if think_match else "No thinking found"
            thinking_length = len(thinking.split())
            
            # Check if thinking is of sufficient length
            if thinking_length >= THINK_LENGTH:
                reward += 0.1  # Small reward for sufficient thinking
            
            # Extract predicted GO terms
            predicted_terms = extract_go_terms_from_response(completion)
            
            # Calculate precision and recall
            true_positives = len(set(predicted_terms) & set(terms))
            precision = true_positives / len(predicted_terms) if predicted_terms else 0
            recall = true_positives / len(terms) if terms else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Reward based on F1 score (higher reward for better predictions)
            reward += GO_PREDICTION_REWARD * f1_score
            
            # Additional reward for properly formatted output with \boxed{}
            if re.search(r'\\boxed{([^}]+)}', completion):
                reward += FORMATTED_OUTPUT_REWARD
            
            # Log metrics for this completion
            wandb.log({
                f"reward/completion_{i}/total_reward": reward,
                f"reward/completion_{i}/precision": precision,
                f"reward/completion_{i}/recall": recall,
                f"reward/completion_{i}/f1_score": f1_score,
                f"reward/completion_{i}/true_positives": true_positives,
                f"reward/completion_{i}/predicted_count": len(predicted_terms),
                f"reward/completion_{i}/ground_truth_count": len(terms),
                f"reward/completion_{i}/thinking_length": thinking_length,
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
            project="protein-rl",
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
    gpu_memory_utilization=0.8,
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
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_generations=4,
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

