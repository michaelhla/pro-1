import json

def main():
    # Reference sequence
    ref_seq = "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSRTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
    
    # Load JSON data
    with open("results/inference/creative_top_performers_with_rmsd.json", "r") as f:
        data = json.load(f)
    
    output_lines = []
    
    for i, item in enumerate(data):
        seq = item["sequence"]
        rmsd = item["rmsd"]
        iteration = item["iteration"]
        
        output_lines.append(f"Sequence {i+1} (Iteration {iteration}, RMSD: {rmsd:.4f})")
        
        # Create aligned sequences with differences in lowercase
        ref_aligned = ""
        new_aligned = ""
        mutations = []
        
        # Handle all possible cases (substitutions, insertions, deletions)
        max_len = max(len(ref_seq), len(seq))
        for j in range(max_len):
            if j < len(ref_seq) and j < len(seq):
                # Both sequences have characters at this position
                if ref_seq[j] == seq[j]:
                    ref_aligned += ref_seq[j]
                    new_aligned += seq[j]
                else:
                    ref_aligned += ref_seq[j].lower()
                    new_aligned += seq[j].lower()
                    mutations.append(f"{ref_seq[j]}{j+1}{seq[j]}")
            elif j < len(ref_seq):
                # Deletion in new sequence
                ref_aligned += ref_seq[j].lower()
                new_aligned += '-'
                mutations.append(f"{ref_seq[j]}{j+1}-")
            else:
                # Insertion in new sequence
                ref_aligned += '-'
                new_aligned += seq[j].lower()
                mutations.append(f"-{j+1}{seq[j]}")
        
        # Output aligned sequences
        output_lines.append(ref_aligned)
        output_lines.append(new_aligned)
        output_lines.append(f"Mutations: {', '.join(mutations)}")
        output_lines.append("")
    
    # Write to file
    with open("sequence_diff.txt", "w") as f:
        f.write("\n".join(output_lines))
    
    print("Sequence diff complete. Results written to sequence_diff.txt")

if __name__ == "__main__":
    main()
