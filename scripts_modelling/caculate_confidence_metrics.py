#!/usr/bin/env python3
import os
import pandas as pd
from Bio.PDB import PDBParser
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import argparse
import glob

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from metrics import (calculate_global_plddt, cdr_plddts, calculate_iptms, calculate_pdockq, calculate_pdockq2, calculate_ipsae_for_seed,
                        remove_low_plddt_and_get_absolute_indices, has_tcr_peptide_contact, get_sync_indices_for_pae,
                        remove_atoms_from_pae_matrix, remove_atoms_from_pde_matrix)
from utils import (cif_to_pdb, merge_pdb)

def process_tcr_folders(base_path, threshold=70, fast=False):
    """
    Processes TCR folders to calculate metrics for each model and write them to a CSV file. If fast mode is enabled and a complete CSV file already exists, it will read metrics from the existing file instead of recalculating them.
    :param base_path: Path to the base folder containing TCR subfolders
    :param threshold: PLDDT threshold for cleaning the model (default is 70)
    :param fast: If True, enables fast mode which reads metrics from an existing CSV file if it exists and is complete, instead of recalculating them (default is False)
    """
    
    if base_path.endswith("/"):
        basename = base_path.rstrip("/").split("/")[-1] 
    else:
        basename = base_path.split("/")[-1]
    
    tcr_id = basename.split("tcr_", 1)[1] if basename.startswith("tcr_") else basename
    folder_path = base_path
    rows = []
     
    # Create the CSV file path for this specific tcr_id
    output_csv = os.path.join(folder_path, f"metrics_{tcr_id}.csv")

    # Search files
    json_input_path = next((f for f in glob.glob(os.path.join(folder_path, "*_data.json"))), None)
    if json_input_path is None:
        print(f"No *_data.json file found in {folder_path}")

    ranking_file_path = next((f for f in glob.glob(os.path.join(folder_path, "*ranking_scores.csv"))), None)
    if not os.path.exists(ranking_file_path):
        print(f"No ranking_scores.csv file found in {folder_path}")
    else:
        ranking_file = ranking_file_path
        ranking_df = pd.read_csv(ranking_file)
        # rename sample to model_number
        ranking_df = ranking_df.rename(columns={"sample": "model_number"})

    cols = ['pdb_id', 'model_number', 'seed', 'plddt', 'cdr1a_plddt', 'cdr1b_plddt', 'cdr2a_plddt', 'cdr2b_plddt',
                            'cdr3a_plddt', 'cdr3b_plddt', 'iptm_mean', 'iptm_tcrpmhc', 'pdockq', 'avgipde_b', 'avgipde_a','avgipae_a',
                            'avgipae_b', 'pdockq2_a', 'pdockq2_b', 'has_contacts', 'ipsae', 'ipsae_d0chn', 'ipsae_d0dom', 'iptm_d0chn', 'lis', 'ranking_score']
    if fast: 
        print(f"Fast mode enabled")
        if os.path.exists(output_csv):
            print(f"CSV file {output_csv} already exists. Metrics will be parsed from this file without recalculating.")
            # If the file exists and has all the necessary columns, we can read the existing metrics into the rows list
            with open(output_csv, mode='r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                # See if all required columns are present
                if not all(col in reader.fieldnames for col in cols):
                    print(f"CSV file {output_csv} is missing some required columns. Fast mode cannot be used with an incomplete metrics file in folder {folder_path}. Running in normal mode instead.")
                    fast = False
                else:
                    for row in reader:
                        rows.append(row)
        else:
            print(f"CSV file {output_csv} does not exist. Fast mode cannot be used without an existing metrics file in folder {folder_path}. Running in normal mode instead.")
            fast = False
    
    # Check fast mode again after verifying the existence of the CSV file
    if not fast:
        if os.path.exists(output_csv):
            os.remove(output_csv)
            print(f"Existing CSV file {output_csv} removed because fast mode is not enabled. Metrics will be recalculated and written to a new CSV file.")

        with open(output_csv, mode='w', newline='') as csvfile:
            fieldnames = cols
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader() 
            
            for seed_folder in os.listdir(folder_path):
                if seed_folder.startswith("seed"):
                    model_number = seed_folder.split("-")[-1]
                    seed_number = seed_folder.split("-")[1].split("_")[0]
                    print(f"\nProcessing model {model_number} in TCR {tcr_id}...\n")
                    seed_folder_path = os.path.join(folder_path, seed_folder)
                    
                    # Initialize variables for file paths
                    model_file_path = None
                    file_output_path = None  
                    summary_json_file_path = None
                    confidence_json_file_path = None
                    merged_pdb = None
                    files = sorted(os.listdir(seed_folder_path))
                    
                    # Identify files in the seed folder
                    for file in files:
                        if file.endswith("model.cif"):
                            model_file_path = os.path.join(seed_folder_path, file)
                            print("Model_file_path", model_file_path)
                            
                            # Remove low PLDDT residues and get removed atom indices
                            file_output_path = os.path.join(seed_folder_path, "model_cleaned.cif")
                            removed_atom_numbers = remove_low_plddt_and_get_absolute_indices(model_file_path, file_output_path, threshold=threshold)
                            print("File output path:", file_output_path)
                            
                            # Convert cleaned CIF to PDB and merge chains
                            cif_to_pdb(file_output_path)
                            pdb_file_path = os.path.splitext(file_output_path)[0] + ".pdb"
                            
                            # Merge chains
                            merge_pdb(pdb_file_path)
                            merged_pdb = os.path.splitext(file_output_path)[0] + "_merged.pdb"
                            
                            # Check for TCR-peptide contacts in the merged PDB structure
                            parser = PDBParser(QUIET=True)
                            structure = parser.get_structure("structure", pdb_file_path)
                            contacts_ok = has_tcr_peptide_contact(structure, cutoff=10.0)
                            print(f"Contacts (A/B to peptide <= 10 A): {contacts_ok}")

                        elif file.endswith("summary_confidences.json"):
                            summary_json_file_path = os.path.join(seed_folder_path, file)
                            print(f"Summary JSON file: {summary_json_file_path}")

                        elif file.endswith("confidences.json"):
                            confidence_json_file_path = os.path.join(seed_folder_path, file)
                            print(f"Confidence JSON file: {confidence_json_file_path}")
                            
                    if confidence_json_file_path and merged_pdb:
                        sorted_removed_indices = get_sync_indices_for_pae(json_input_path, pdb_file_path)
                        
                        # Generate new PAE and PDE matrices with removed residues filtered out, and save them in the seed folder
                        pae_output_path = os.path.join(folder_path, f"pae_{model_number}.npy")
                        pde_output_path = os.path.join(folder_path, f"pde_{model_number}.npy")

                        pae_matrix = remove_atoms_from_pae_matrix(confidence_json_file_path, sorted_removed_indices, pae_output_path)
                        pde_matrix = remove_atoms_from_pde_matrix(confidence_json_file_path, sorted_removed_indices, pde_output_path)

                        # Calculate all metrics
                        global_plddt = calculate_global_plddt(file_output_path)
                        cdr1a_plddt, cdr2a_plddt, cdr3a_plddt, cdr1b_plddt, cdr2b_plddt, cdr3b_plddt = cdr_plddts(file_output_path, "D", "E")
                        iptm_mean, iptm_tcrpmhc = calculate_iptms(summary_json_file_path)
                        _, pdockq = calculate_pdockq(merged_pdb)
                        _, avgipde_A, avgipde_B, avgipae_A, avgipae_B, pdockq2_A, pdockq2_B = calculate_pdockq2(merged_pdb, pde_output_path, pae_output_path)
                        ipsae, ipsae_d0chn, ipsae_d0dom, iptm_d0chn, lis = calculate_ipsae_for_seed(seed_folder_path, model_file_path, confidence_json_file_path, scripts_dir="./")

                        print(f"\nMetrics calculated for model {model_number} ---------------------------------------- ")        
                        print(f"Global PLDDT: {global_plddt}")
                        print(f"CDR1s PLDDT: CDR1a {cdr1a_plddt}, CDR1b {cdr1b_plddt}, CDR2a {cdr2a_plddt}, CDR2b {cdr2b_plddt}, CDR3a {cdr3a_plddt}, CDR3b {cdr3b_plddt}")
                        print(f"Chain IPTM mean: {iptm_mean}")
                        print(f"Interface TCR-pMHC IPTM mean: {iptm_tcrpmhc}")
                        print(f"pDockQ: {pdockq}")
                        print(f"Average iPDE A: {avgipde_A}, Average iPDE B: {avgipde_B}")
                        print(f"Average iPAE A: {avgipae_A}, Average iPAE B: {avgipae_B}")
                        print(f"pDockQ2 A: {pdockq2_A}, pDockQ2 B: {pdockq2_B}")
                        print(f"ipSAE metrics ipSAE: {ipsae}, ipSAE d0chn {ipsae_d0chn}, ipSAE d0dom {ipsae_d0dom}, IPTM d0chn {iptm_d0chn}, LIS {lis}")
                        print("------------------------------------------------------------------------")
                        
                        # find the ranking_score for this model_number in the ranking_scores.csv file only one value
                        ranking_score = ranking_df.loc[ranking_df['model_number'] == int(model_number), 'ranking_score'].values
                        row_data = {
                            'pdb_id': tcr_id,
                            'model_number': model_number,
                            'seed': seed_number,
                            'plddt': global_plddt,
                            'cdr1a_plddt': cdr1a_plddt,
                            'cdr1b_plddt': cdr1b_plddt,
                            'cdr2a_plddt': cdr2a_plddt,
                            'cdr2b_plddt': cdr2b_plddt,
                            'cdr3a_plddt': cdr3a_plddt,
                            'cdr3b_plddt': cdr3b_plddt,
                            'iptm_mean': iptm_mean,
                            'iptm_tcrpmhc': iptm_tcrpmhc,
                            'pdockq': pdockq,
                            'avgipde_a': avgipde_A,
                            'avgipde_b': avgipde_B,
                            'avgipae_a': avgipae_A,
                            'avgipae_b': avgipae_B,
                            'pdockq2_a': pdockq2_A,
                            'pdockq2_b': pdockq2_B,
                            'has_contacts': contacts_ok,
                            'ipsae': ipsae,
                            'ipsae_d0chn': ipsae_d0chn,
                            'ipsae_d0dom': ipsae_d0dom,
                            'iptm_d0chn': iptm_d0chn,
                            'lis': lis,
                            'ranking_score': ranking_score[0] if len(ranking_score) > 0 else None
                        }

                        # Append to list
                        rows.append(row_data)

                        # Write to CSV
                        writer.writerow(row_data)

                        # Remove all temporary/generated files
                        files_to_remove = [merged_pdb, pae_output_path, pde_output_path, file_output_path, pdb_file_path]
                        for f in files_to_remove:
                            if f and os.path.exists(f):
                                os.remove(f)

                        # Remove any .txt or .pml files in the seed folder
                        for ext in ["*.txt", "*.pml"]:
                            for temp_file in glob.glob(os.path.join(seed_folder_path, ext)):
                                os.remove(temp_file)
                    else:
                        print("Error processing metrics")

    metrics_df = pd.DataFrame(rows)
    return metrics_df

def process_tcr_folder_parallel(folder_path, threshold=70, fast=False):
    """
    Process a single TCR folder (subfolder) and return a DataFrame with metrics.
    Removes all temporary files after processing.
    """
    try:
        metrics_df = process_tcr_folders(folder_path, threshold=threshold,fast=fast)
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
        metrics_df = pd.DataFrame()

    return metrics_df


def main():
    parser = argparse.ArgumentParser(description="Process TCR folders and calculate metrics in parallel.")
    parser.add_argument("base_path", type=str, help="Path to the main TCR directory.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output csv")
    parser.add_argument("--threshold", type=int, default=70, help="PLDDT threshold for filtering residues.")
    parser.add_argument("--fast", action="store_true", default=False, help="If set, skips recalculating metrics and reads from existing CSV files if available.")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPUs to use for parallel processing.")
    args = parser.parse_args()

    tcr_subfolders = [os.path.join(args.base_path, f) for f in os.listdir(args.base_path)
                      if os.path.isdir(os.path.join(args.base_path, f))]

    all_metrics = []

    # Parallel execution
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_tcr_folder_parallel, folder, args.threshold, args.fast): folder for folder in tcr_subfolders}
        for future in as_completed(futures):
            folder = futures[future]
            try:
                df = future.result()
                all_metrics.append(df)
                print(f"Completed processing: {folder}")
            except Exception as e:
                print(f"Error in parallel processing for folder {folder}: {e}")

    # Concatenate all metrics
    if all_metrics:
        full_metrics_df = pd.concat(all_metrics, ignore_index=True)
        full_metrics_df.to_csv(args.output, index=False)
        print(f"Metrics saved to {args.output}")
    else:
        print("No metrics to save.")


    print("Processing complete.")


if __name__ == "__main__":
    main()