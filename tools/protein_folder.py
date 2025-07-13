#!/usr/bin/env python3
"""
Protein Structure Prediction Tool using ESMFold

Simple tool that folds protein sequences using ESMFold and saves them as PDB files.
"""

import os
import torch
import time
from typing import Optional
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37


class ProteinFolder:
    """
    Simple protein folder using ESMFold - based on stability_reward.py
    """
    
    def __init__(self, protein_model_path="facebook/esmfold_v1", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.protein_model = self._load_protein_model(protein_model_path)
        self.cached_structures = {}

    def _load_protein_model(self, model_path):
        """Load ESMFold model for structure prediction"""
        start_time = time.time()
        local_path = 'model_cache/'
        if os.path.exists(local_path):
            model = EsmForProteinFolding.from_pretrained(local_path)
        else:
            model = EsmForProteinFolding.from_pretrained(model_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            model.save_pretrained(local_path)

        # Move model to specified device
        model = model.to(self.device)
        print(f"ESM loading took {time.time() - start_time:.2f} seconds")
        return model

    def predict_structure(self, sequence, uniprot_id=None):
        """Predict protein structure using ESMFold"""
        if sequence in self.cached_structures:
            return self.cached_structures[sequence]

        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        tokenized_input = tokenizer(
            [sequence], 
            return_tensors="pt", 
            add_special_tokens=False
        )['input_ids'].to(self.device)

        with torch.no_grad():
            output = self.protein_model(tokenized_input)

        pdb_file = self.convert_outputs_to_pdb(output, uniprot_id)[0]

        # Cache the result
        self.cached_structures[sequence] = pdb_file
        print(f"Structure prediction took {time.time() - start_time:.2f} seconds")
        return pdb_file

    def convert_outputs_to_pdb(self, outputs, uniprot_id=None):
        final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
        outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = outputs["atom37_atom_exists"]
        pdbs = []

        for i in range(outputs["aatype"].shape[0]):
            aa = outputs["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = outputs["residue_index"][i] + 1
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=outputs["plddt"][i],
                chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
            )
            pdbs.append(to_pdb(pred))

        output_dir = "predicted_structures"
        os.makedirs(output_dir, exist_ok=True)

        pdb_files = []
        for i, pdb in enumerate(pdbs):
            if uniprot_id:
                pdb_path = os.path.join(output_dir, f"{uniprot_id}.pdb")
            else:
                pdb_path = os.path.join(output_dir, f"{i}.pdb")
            with open(pdb_path, "w") as f:
                f.write(pdb)
            pdb_files.append(pdb_path)
        return pdb_files


# Global instance
_protein_folder = None


def get_protein_folder():
    """Get or create the global protein folder instance."""
    global _protein_folder
    if _protein_folder is None:
        _protein_folder = ProteinFolder()
    return _protein_folder


def fold_protein(sequence: str, protein_id: Optional[str] = None) -> str:
    """
    Fold a protein sequence and return the path to the PDB file.
    
    Args:
        sequence: Amino acid sequence to fold
        protein_id: Optional identifier for the protein
        
    Returns:
        Path to the saved PDB file
    """
    folder = get_protein_folder()
    return folder.predict_structure(sequence, protein_id)


if __name__ == "__main__":
    # Example usage
    folder = ProteinFolder()
    
    # Example carbonic anhydrase sequence (partial)
    ca_sequence = "SHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
    
    try:
        pdb_path = folder.predict_structure(ca_sequence, "carbonic_anhydrase_example")
        print(f"Successfully folded carbonic anhydrase structure: {pdb_path}")
        
    except Exception as e:
        print(f"Error: {e}") 