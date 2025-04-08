import json

def extract_sequences():
    # Read the JSON file
    with open("results/inference/top_performers_with_rmsd.json", "r") as f:
        data = json.load(f)
    
    # Extract sequences and write to file
    with open("sequences.txt", "w") as f:
        # Write reference sequence first
        f.write("Reference sequence:\n")
        f.write("MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSRTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK")
        f.write("\n\nVariant sequences:\n")
        
        # Write each variant sequence with its iteration number and RMSD
        for i, item in enumerate(data, 1):
            f.write(f"\nSequence {i} (Iteration {item['iteration']}, RMSD: {item['rmsd']:.4f}):\n")
            f.write(f"{item['sequence']}\n")

if __name__ == "__main__":
    extract_sequences() 