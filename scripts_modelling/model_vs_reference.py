#!/usr/bin/env python3
import os
import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from utils import merge_pdb, cif_to_pdb
from rmsd import run_dockq, calculate_rmsd

def parse_general_file(general_file):
    """
    Parses the general file and creates a dictionary mapping PDB IDs to specific chain information
    such as 'tcra_chain', 'tcrb_chain', 'peptide_chain', and 'mhc_chain'.
    
    :param general_file: Path to the general file.
    :return: A dictionary where keys are PDB IDs and values are dictionaries with chain information
    """
    df = pd.read_csv(general_file, sep='\t')
    pdb_dict = {}

    for pdb_id, group in df.groupby('pdb.id'):
        pdb_id = pdb_id.split('.')[0]  
        chains = {
            'tcra_chain': None,
            'tcrb_chain': None,
            'peptide_chain': None,
            'mhc_chain': None}

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

def classify_model(row):
    irmsd = row['TCR-iRMSD']
    dockq = row['DockQ']
    
    if irmsd < 2 and dockq > 0.8:
        return 'HQ'
    elif irmsd < 2 or (irmsd < 5 and dockq >= 0.49):
        return 'MQ'
    elif irmsd < 5 or dockq >= 0.23:
        return 'AQ'
    else:
        return 'LQ'

def process_pdbs(reference_file_path, model_folder_path, chain_dict):
    dataframe = pd.DataFrame()
    pdb_file_path = os.path.basename(reference_file_path)
    pdb_id = pdb_file_path.split(".")[0]
    print(f"Processing structure for pdb_id: {pdb_id}, {reference_file_path}...")
    print(chain_dict[pdb_id])
    merge_pdb(reference_file_path, chain_mapping = chain_dict[pdb_id])
    dockq_native = os.path.splitext(reference_file_path)[0] + "_merged.pdb"
    model_file_paths = []
    for seed_folder in os.listdir(model_folder_path):
        if seed_folder.startswith("seed"):
            seed_folder_path = os.path.join(model_folder_path, seed_folder)
            files = sorted(os.listdir(seed_folder_path))
            for file in files:
                if file.endswith("model.cif"):
                    model_file_path = os.path.join(seed_folder_path, file)
                    model_file_paths.append(model_file_path)

    print(f"Found {len(model_file_paths)} model files for {pdb_id}, processing each model...")
    model_file_paths = sorted (model_file_paths)
    for model_file in model_file_paths:
        cif_to_pdb(model_file)
        pdb_model_file_path = os.path.splitext(model_file)[0] + ".pdb"
        merge_pdb(pdb_model_file_path)
        dockq_model = os.path.splitext(pdb_model_file_path)[0] + "_merged.pdb"
        model_number = model_file.split("/")[4].split("-")[-1]
        print(f"Processing model {model_number}...")
        rmsd_result = calculate_rmsd(reference_file_path, pdb_model_file_path, pdb_id, chain_dict, distance_cutoff=10.0)
        if isinstance(rmsd_result, tuple) and len(rmsd_result) == 7:
            result_string, overall_rmsd, rmsd_TCRA_TCRB, rmsd_Peptide, rmsd_MHC_B2M, rmsd_CDR_TCRA, rmsd_CDR_TCRB = rmsd_result
        else:
            result_string = str(rmsd_result)
            overall_rmsd = rmsd_TCRA_TCRB = rmsd_Peptide = rmsd_MHC_B2M = rmsd_CDR_TCRA = rmsd_CDR_TCRB = None
            print(f"Error calculating RMSD for {pdb_model_file_path}: {result_string}")
        print(dockq_model, dockq_native)
        dockq_score, irmsd, lrmsd, fnat, clashes = run_dockq(dockq_model, dockq_native)
        
        print("\n----------------------")
        print(f"Processed: {pdb_id} model {model_number}")
        print("----------------------")
        print(result_string) 
        print("DockQ results:")
        print(f"DockQ score: {dockq_score:.3f}, iRMSD: {irmsd:.3f}, LRMSD: {lrmsd:.3f}, FNAT: {fnat:.3f}, Clashes: {clashes}")
        print("----------------------\n")    

        row = {"pdb_id": pdb_id,
                "model_number": model_number,
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
                "Clashes": clashes}
        row["Quality"] = classify_model(row)
        
        dataframe = pd.concat([dataframe, pd.DataFrame([row])], ignore_index=True)

        for f in [pdb_model_file_path, dockq_model]:
            if os.path.exists(f):
                os.remove(f)
    
    merged_ref = os.path.splitext(reference_file_path)[0] + "_merged.pdb"
    if os.path.exists(merged_ref):
        os.remove(merged_ref)     

    return dataframe

def process_wrapper(pdb_file, reference_dir, model_superdir, chain_dict):
    pdb_path = os.path.join(reference_dir, pdb_file)
    pdb_id = os.path.splitext(pdb_file)[0]
    model_folder_path = os.path.join(model_superdir, pdb_id)
    print(f"\n=== Processing {pdb_id} ===")
    try:
        df = process_pdbs(pdb_path, model_folder_path, chain_dict)
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        df = pd.DataFrame()  # Retornar vacío si falla
    return df

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Process TCR folders to clean CIF files and update PAE matrices.")
    parser.add_argument("--reference_dir", type=str, help="Path to the TCR-pMHC reference directory.")
    parser.add_argument("--model_superdir", type=str, help="Path to the TCR-pMHC model directory with AF3 output folders.")
    parser.add_argument("--output", type=str, help="Path to the output CSV")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPUs to use for parallel processing")
    args = parser.parse_args()

    # Cargar diccionario de cadenas
    chain_dict = parse_general_file('../data/structures_annotation/general.txt')

    pdb_files = [f for f in os.listdir(args.reference_dir) if f.endswith(".pdb")]
    general_df = pd.DataFrame()

    # Paralelización
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_wrapper, pdb_file, args.reference_dir, args.model_superdir, chain_dict) 
                   for pdb_file in pdb_files]
        for future in as_completed(futures):
            df_result = future.result()
            general_df = pd.concat([general_df, df_result], ignore_index=True)

    # Guardar CSV
    general_df.to_csv(args.output, index=False)
    print(f"\nAll processed metrics saved to {args.output}")

if __name__ == "__main__":
    main()