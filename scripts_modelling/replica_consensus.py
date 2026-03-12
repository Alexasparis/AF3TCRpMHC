#!/usr/bin/env python3
import os
import pandas as pd
import argparse
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from utils import merge_pdb, cif_to_pdb
from rmsd import run_dockq, calculate_rmsd

# --------------------------
# Parse chain information
# --------------------------
def parse_general_file(general_file):
    df = pd.read_csv(general_file, sep='\t')
    pdb_dict = {}

    for pdb_id, group in df.groupby('pdb.id'):
        pdb_id = pdb_id.split('.')[0]
        chains = {
            'tcra_chain': None,
            'tcrb_chain': None,
            'peptide_chain': None,
            'mhc_chain': None,
            'b2_chain': None
        }
        for _, row in group.iterrows():
            chain_id = row['chain.id']
            chain_type = row['chain.type']
            chain_component = row['chain.component']
            chain_supertype = row['chain.supertype']

            if chain_component == 'TCR' and chain_type == 'TRA':
                chains['tcra_chain'] = chain_id
            elif chain_component == 'TCR' and chain_type == 'TRB':
                chains['tcrb_chain'] = chain_id
            elif chain_component == 'PEPTIDE':
                chains['peptide_chain'] = chain_id
            elif chain_component == 'MHC' and chain_supertype == 'MHCI' and chain_type == 'MHCa':
                chains['mhc_chain'] = chain_id
            elif chain_component == 'MHC' and chain_supertype == 'MHCI' and chain_type == 'MHCb':
                chains['b2_chain'] = chain_id

        pdb_dict[pdb_id] = chains
    return pdb_dict

# --------------------------
# Pairwise RMSD + DockQ
# --------------------------
def process_pdbs_pairwise(model_folder_path, pdb_id, chain_dict):
    dataframe = pd.DataFrame()
    model_file_paths = []

    # Collect all model (.cif) files in seed* folders
    for seed_folder in os.listdir(model_folder_path):
        if seed_folder.startswith("seed"):
            seed_folder_path = os.path.join(model_folder_path, seed_folder)
            for file in sorted(os.listdir(seed_folder_path)):
                if file.endswith("model.cif"):
                    model_file_paths.append(os.path.join(seed_folder_path, file))

    print(f"Found {len(model_file_paths)} model files, processing pairwise comparisons...")

    # Convert .cif to .pdb and merge for DockQ
    pdb_model_paths = []
    merged_model_paths = []
    for model_file in sorted(model_file_paths):
        try:
            cif_to_pdb(model_file)
            pdb_model_file_path = os.path.splitext(model_file)[0] + ".pdb"
            pdb_model_paths.append(pdb_model_file_path)

            merge_pdb(pdb_model_file_path)
            merged_model_file_path = os.path.splitext(pdb_model_file_path)[0] + "_merged.pdb"
            merged_model_paths.append(merged_model_file_path)
        except Exception as e:
            print(f"Error processing model {model_file}: {e}")

    # Pairwise comparisons
    for i, j in combinations(range(len(pdb_model_paths)), 2):
        pdb_a = pdb_model_paths[i]
        pdb_b = pdb_model_paths[j]
        merged_a = merged_model_paths[i]
        merged_b = merged_model_paths[j]

        model_a_num = os.path.basename(pdb_a).split("-")[-1].split(".")[0]
        model_b_num = os.path.basename(pdb_b).split("-")[-1].split(".")[0]

        print(f"Processing pair: model {model_a_num} vs model {model_b_num}...")

        # RMSD from raw PDBs
        try:
            rmsd_result = calculate_rmsd(pdb_a, pdb_b, pdb_id=pdb_id, chain_dict=chain_dict, distance_cutoff=10.0, model_pairwise=True)
            if isinstance(rmsd_result, tuple) and len(rmsd_result) == 7:
                result_string, overall_rmsd, rmsd_TCRA_TCRB, rmsd_Peptide, rmsd_MHC_B2M, rmsd_CDR_TCRA, rmsd_CDR_TCRB = rmsd_result
            else:
                result_string = str(rmsd_result)
                overall_rmsd = rmsd_TCRA_TCRB = rmsd_Peptide = rmsd_MHC_B2M = rmsd_CDR_TCRA = rmsd_CDR_TCRB = None
                print(f"RMSD calculation issue for pair {model_a_num}-{model_b_num}: {result_string}")
        except Exception as e:
            print(f"RMSD failed for pair {model_a_num}-{model_b_num}: {e}")
            overall_rmsd = rmsd_TCRA_TCRB = rmsd_Peptide = rmsd_MHC_B2M = rmsd_CDR_TCRA = rmsd_CDR_TCRB = None

        # DockQ from merged PDBs
        try:
            dockq_score, irmsd, lrmsd, fnat, clashes = run_dockq(merged_a, merged_b)
        except Exception as e:
            print(f"DockQ failed for pair {model_a_num}-{model_b_num}: {e}")
            dockq_score = irmsd = lrmsd = fnat = clashes = None
            
        print("\n----------------------")
        print(f"Processed: {pdb_id} pair {model_a_num} vs {model_b_num}")
        print("----------------------")
        print(result_string)  # RMSD detailed string if available
        print("DockQ results:")
        print(f"DockQ score: {dockq_score:.3f}, iRMSD: {irmsd:.3f}, LRMSD: {lrmsd:.3f}, FNAT: {fnat:.3f}, Clashes: {clashes}")
        print("----------------------\n")

        row = {
            "model_number_1": model_a_num,
            "model_number_2": model_b_num,
            "iRMSD": overall_rmsd,
            "TCR-iRMSD": rmsd_TCRA_TCRB,
            "Peptide-RMSD": rmsd_Peptide,
            "MHC-iRMSD": rmsd_MHC_B2M,
            "CDR3a-RMSD": rmsd_CDR_TCRA,
            "CDR3b-RMSD": rmsd_CDR_TCRB,
            "DockQ": dockq_score,
            "iRMSD-DockQ": irmsd,
            "lRMSD-DockQ": lrmsd,
            "Fnat": fnat,
            "Clashes": clashes
        }

        dataframe = pd.concat([dataframe, pd.DataFrame([row])], ignore_index=True)

    # Cleanup
    for f in pdb_model_paths + merged_model_paths:
        if os.path.exists(f):
            os.remove(f)

    return dataframe

# --------------------------
# Wrapper for parallel processing
# --------------------------
def process_wrapper(pdb_file, model_superdir, chain_dict):
    pdb_id = os.path.splitext(pdb_file)[0]
    model_folder_path = os.path.join(model_superdir, pdb_id)
    print(f"\n=== Processing {pdb_id} ===")
    try:
        df = process_pdbs_pairwise(model_folder_path, pdb_id, chain_dict)
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        df = pd.DataFrame()
    return df

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Process TCR model folders and calculate pairwise RMSD and DockQ."
    )
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the TCR-pMHC model directory.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output CSV.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    args = parser.parse_args()

    # Load chain info
    chain_dict = parse_general_file("../data/structures_annotation/general.txt")

    # List PDB folders
    pdb_files = [f for f in os.listdir(args.model_dir) if os.path.isdir(os.path.join(args.model_dir, f))]
    print(f"Found {len(pdb_files)} PDB folders to process.")

    general_df = pd.DataFrame()

    # Parallel processing
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_wrapper, pdb_file, args.model_dir, chain_dict) for pdb_file in pdb_files]
        for future in as_completed(futures):
            df_result = future.result()
            general_df = pd.concat([general_df, df_result], ignore_index=True)

    # Save output
    general_df.to_csv(args.output, index=False)
    print(f"\nAll processed metrics saved to {args.output}")


if __name__ == "__main__":
    main()