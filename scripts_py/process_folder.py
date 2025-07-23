#!/usr/bin/env python3

import os
from Bio.PDB import MMCIFParser, MMCIFIO
from Bio import PDB
import numpy as np
import json
import subprocess
from statistics import mean
import re
import csv
import argparse
import pandas as pd

residue_mapping = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

def remove_low_plddt_and_get_absolute_indices(cif_file, output_file, threshold=50):
    """
    Removes low PLDDT residues from the N- and C-terminal ends of each chain in the CIF file,
    tracks the indices of the removed residues, and returns a list of absolute indices
    of the removed residues.

    :param cif_file: Path to the input CIF file.
    :param output_file: Path to save the cleaned CIF file.
    :param threshold: PLDDT threshold for filtering residues.
    :return: List of absolute indices of the removed residues.
    """
    # Suppress warnings and parse the CIF file
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_file)

    # Get the length of each chain
    chain_lengths = {chain.id: len(list(chain.get_residues())) for chain in structure[0]}

    # Dictionary to track removed residue indices for each chain
    removed_indices = {chain.id: {"N": [], "C": []} for chain in structure[0]}

    # Process the removal of low PLDDT residues
    model = structure[0]
    for chain in model:
        residues = list(chain.get_residues())
        chain_id = chain.id
        
        # N-terminal trimming
        while residues:
            first_residue = residues[0]
            plddts = [atom.bfactor for atom in first_residue.get_atoms()]
            if any(plddt < threshold for plddt in plddts):  # If any atom has PLDDT < threshold
                removed_indices[chain_id]["N"].append(first_residue.id[1])  # Track residue index
                chain.detach_child(first_residue.id)  # Remove residue
                residues.pop(0)  # Update residue list
            else:
                break
        
        # C-terminal trimming
        while residues:
            last_residue = residues[-1]
            plddts = [atom.bfactor for atom in last_residue.get_atoms()]
            if any(plddt < threshold for plddt in plddts):  # If any atom has PLDDT < threshold
                removed_indices[chain_id]["C"].append(last_residue.id[1])  # Track residue index
                chain.detach_child(last_residue.id)  # Remove residue
                residues.pop()  # Update residue list
            else:
                break

    # Save the cleaned CIF file
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(output_file)

    # List to hold absolute indices of removed residues
    absolute_indices = []
    global_residue_number = 1  # Start the global residue index

    # Calculate absolute indices for removed residues
    for chain, residues in removed_indices.items():
        chain_length = chain_lengths.get(chain, 0)

        # Lists for N-terminal and C-terminal residues that were removed
        n_terminal_residues = residues.get("N", [])
        c_terminal_residues = residues.get("C", [])

        # Filter out residues that are valid within the chain length
        n_terminal_residues = [residue for residue in n_terminal_residues if residue <= chain_length]
        c_terminal_residues = [residue for residue in c_terminal_residues if residue <= chain_length]

        # Calculate absolute indices for removed residues and add to the list
        absolute_indices.extend([global_residue_number + residue - 1 for residue in sorted(n_terminal_residues)])
        absolute_indices.extend([global_residue_number + residue - 1 for residue in sorted(c_terminal_residues)])

        # Update global residue index for the next chain
        global_residue_number += chain_length

    # Return the list of absolute indices of removed residues
    return absolute_indices

def remove_atoms_from_pae_matrix(json_file, removed_atom_numbers, output_file):
    """
    Removes atoms from the PAE matrix in a JSON file based on the atom numbers
    that were removed in the structure (from the CIF file) and saves the updated
    PAE matrix to a NumPy .npy file.

    :param json_file: Path to the original JSON file containing the PAE matrix.
    :param removed_atom_numbers: List of atom numbers (1-based) that were removed.
    :param output_file: Path to the output NumPy .npy file where the updated PAE matrix will be saved.
    """
    # Load the PAE matrix from the original JSON file
    with open(json_file, 'r') as f:
        pae_data = json.load(f)

    # Convert atom numbers to 0-based indices
    removed_atom_indices = [atom_num - 1 for atom_num in removed_atom_numbers]

    # Get the PAE matrix from the JSON data
    pae_matrix = np.array(pae_data['pae'])

    # Remove corresponding rows and columns in the PAE matrix
    pae_matrix = np.delete(pae_matrix, removed_atom_indices, axis=0)
    pae_matrix = np.delete(pae_matrix, removed_atom_indices, axis=1)

    # Save the updated PAE matrix as a NumPy .npy file in binary format
    np.save(output_file, pae_matrix)
    return pae_matrix

def cif_to_pdb(cif_file):
    """
    Converts a single CIF file to PDB format and saves it in the same folder with the same name.
    
    Parameters:
    - cif_file: str, path to the input CIF file.
    """
    # Define the output PDB file path (same folder, same name, .pdb extension)
    pdb_file = os.path.splitext(cif_file)[0]
    beem_path = "../../data_augmentation/BeEM/BeEM"#"/gpfs/projects/bsc72/aascunce/data_augmentation/BeEM/BeEM"
    # Build the PyMOL command to convert CIF to PDB
    command = f"{beem_path} -p={pdb_file} {cif_file}"

    # Suppress the output of the pymol command by redirecting stdout and stderr to os.devnull
    try:
        with open(os.devnull, 'w') as devnull:
            subprocess.run(command, shell=True, stdout=devnull, stderr=devnull, check=True)
        print(f"Successfully converted {cif_file} to {pdb_file}")
    except subprocess.CalledProcessError as e:
        # Handle errors during the conversion process
        print(f"Error converting {cif_file}: {e}")

def merge_pdb(pdb_file):
    """
    Processes a single PDB file, modifies chain IDs, and merges the results into a single PDB file.

    Parameters:
    - pdb_file: str, path to the input PDB file.

    Output:
    - A merged PDB file with the same name as the input, appended with '_merged', saved in the same directory.
    """
    # Define chain IDs
    tcra_id = "D"
    tcrb_id = "E"
    mhc_id = "A"
    b2_id = "B"
    epitope_id = "C"

    # Extract base name and define output file path
    base_name = pdb_file.rsplit(".", 1)[0]  # Name without extension
    output_file_path = f"{base_name}_merged.pdb"  # Add '_merged' to the output file name

    # Preprocess the input file to remove invalid lines and save a temporary cleaned file
    cleaned_pdb_file = f"{base_name}_cleaned.pdb"
    cleaned_lines = remove_headers(pdb_file)
    with open(cleaned_pdb_file, 'w') as cleaned_file:
        cleaned_file.writelines(cleaned_lines)

    # Construct shell commands
    command_AB = (
        f"pdb_selchain -{tcra_id},{tcrb_id} {cleaned_pdb_file} "
        f"| pdb_chain -B | pdb_reres -1 | pdb_delhetatm > B.pdb"
    )
    command_MB = (
        f"pdb_selchain -{mhc_id},{b2_id},{epitope_id} {cleaned_pdb_file} "
        f"| pdb_chain -A | pdb_reres -1 | pdb_delhetatm > A.pdb"
    )

    try:
        # Execute shell commands to generate temporary files
        subprocess.run(command_MB, shell=True, check=True)
        subprocess.run(command_AB, shell=True, check=True)

        # Remove headers and merge the files
        A_lines = remove_headers("A.pdb")
        B_lines = remove_headers("B.pdb")

        # Save the merged PDB file
        with open(output_file_path, 'w') as outfile:
            outfile.writelines(A_lines)
            outfile.writelines(B_lines)

        # Clean up temporary files
        os.remove('A.pdb')
        os.remove('B.pdb')
        os.remove(cleaned_pdb_file)

    except subprocess.CalledProcessError as e:
        print(f"Error processing {pdb_file}: {e}")

def remove_headers(file_path):
    """
    Remove non-ATOM/HETATM lines from a PDB file and validate line length.

    Parameters:
    - file_path: str, path to the PDB file.

    Returns:
    - List of cleaned lines (ATOM/HETATM and properly formatted).
    """
    cleaned_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            # Keep only ATOM/HETATM lines with sufficient length
            if (line.startswith("ATOM") or line.startswith("HETATM")) and len(line) > 21:
                cleaned_lines.append(line)
    return cleaned_lines

def calculate_global_plddt(cif_file_path):
    """
    Calculate the mean of all B-factors in a CIF file using MMCIFParser.

    Args:
        cif_file_path (str): Path to the CIF file.

    Returns:
        float: The mean B-factor, or None if no B-factors are found.
    """
    try:
        # Parse the CIF file
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("structure", cif_file_path)

        # Extract all B-factors
        b_factors = [
            atom.get_bfactor() for model in structure
            for chain in model
            for residue in chain
            for atom in residue
        ]

        if not b_factors:
            print("No B-factors found in the CIF file.")
            return None

        # Calculate the mean B-factor
        mean_b_factor = mean(b_factors)
        return mean_b_factor

    except FileNotFoundError:
        print(f"Error: The file {cif_file_path} was not found.")
        return None
    except Exception as e:
        print(f"Error processing the CIF file: {e}")
        return None
    
def extract_b_factors(cdr_atoms, chain):
    """
    Extract B-factors for a given list of atoms from a specific chain.
    
    Args:
        cdr_atoms: List of tuples containing (atom_name, residue_id, residue_name, chain_id).
        chain: Bio.PDB.Chain object corresponding to the chain to extract B-factors from.
        
    Returns:
        A list of B-factors for the specified atoms.
    """
    b_factors = []
    for atomname, resid, resname, chainid in cdr_atoms:
        if chainid == chain.id:
            try:
                residue = chain[resid]  # Access the residue using its ID
                if atomname in residue:  # Check if the atom exists in the residue
                    atom = residue[atomname]
                    b_factors.append(atom.get_bfactor())
                else:
                    print(f"Atom {atomname} not found in residue {resid} ({resname}) of chain {chain.id}")
            except KeyError:
                print(f"Residue {resid} ({resname}) not found in chain {chain.id}")
    return b_factors

def extract_sequences(pdb_file):
    """
    Extract sequences for all chains from a PDB file in two forms:
    - A dictionary of sequences as single-letter residue codes (string).
    - A dictionary of sequences as lists of (resname, resid) tuples.
    
    Returns:
    - sequences_str (dict): A dictionary with chain_id as key and sequence as string of 1-letter codes.
    - sequences_tuples (dict): A dictionary with chain_id as key and sequence as list of tuples (resname, resid).
    """
    parser=MMCIFParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    sequences_str = {}
    sequences_tuples = {}

    for model in structure:
        for chain in model.get_chains():
            chain_id = chain.get_id()
            sequence_str = []  # For single-letter sequence
            sequence_tuples = []  # For (resname, resid) tuples
            for residue in chain:
                if PDB.is_aa(residue):  # Ensure the residue is an amino acid
                    res_name = residue.get_resname()  
                    resid = residue.get_id()[1]  # Residue ID
                    # Add the single-letter residue code to the string
                    sequence_str.append(residue_mapping.get(res_name, 'X'))  # 'X' if unknown residue
                    # Store the (resname, resid) tuple for residue identity
                    sequence_tuples.append((res_name, resid))  
            sequences_str[chain_id] = ''.join(sequence_str)  # Join into string
            sequences_tuples[chain_id] = sequence_tuples  # Store the tuples
    return sequences_str, sequences_tuples

def cdr_plddts(model_file, alpha_chain, beta_chain):
    model_sequences, model_dict = extract_sequences(model_file)
    residues_A = extract_residues_and_resids(model_file, alpha_chain)
    residues_B = extract_residues_and_resids(model_file, beta_chain)
    anarci_A = run_anarci(model_sequences[alpha_chain])
    anarci_B = run_anarci(model_sequences[beta_chain])
    parsed_A = parse_anarci_output(anarci_A)
    parsed_B = parse_anarci_output(anarci_B)
    map_A = map_imgt_to_original(parsed_A, residues_A)
    map_B = map_imgt_to_original(parsed_B, residues_B)

    # Parse CDR regions from the maps
    cdr3_A = parse_CDR3(map_A)
    cdr3_B = parse_CDR3(map_B)
    cdr2_A = parse_CDR2(map_A)
    cdr2_B = parse_CDR2(map_B)
    cdr1_A = parse_CDR1(map_A)
    cdr1_B = parse_CDR1(map_B)

    # Extract atom information for each CDR
    cdr3_atoms_A = extract_atoms_for_cdr(cdr3_A, model_file, alpha_chain)
    cdr3_atoms_B = extract_atoms_for_cdr(cdr3_B, model_file, beta_chain)
    cdr2_atoms_A = extract_atoms_for_cdr(cdr2_A, model_file, alpha_chain)
    cdr2_atoms_B = extract_atoms_for_cdr(cdr2_B, model_file, beta_chain)
    cdr1_atoms_A = extract_atoms_for_cdr(cdr1_A, model_file, alpha_chain)
    cdr1_atoms_B = extract_atoms_for_cdr(cdr1_B, model_file, beta_chain)

    
    # Parse the structure based on file type (PDB or MMCIF)
    if model_file.endswith(".pdb"):
        parser = PDB.PDBParser(QUIET=True)
    else:
        parser = PDB.MMCIFParser(QUIET=True)
        
    structure = parser.get_structure("Model", model_file)
    chain_A = structure[0][alpha_chain]
    chain_B = structure[0][beta_chain]
    
    # Extract B-factors for each CDR region separately
    b_factors_cdr1_A = extract_b_factors(cdr1_atoms_A, chain_A)
    b_factors_cdr2_A = extract_b_factors(cdr2_atoms_A, chain_A)
    b_factors_cdr3_A = extract_b_factors(cdr3_atoms_A, chain_A)
    
    b_factors_cdr1_B = extract_b_factors(cdr1_atoms_B, chain_B)
    b_factors_cdr2_B = extract_b_factors(cdr2_atoms_B, chain_B)
    b_factors_cdr3_B = extract_b_factors(cdr3_atoms_B, chain_B)

    # Mean 
    mean_cdr1_A = np.mean(b_factors_cdr1_A)
    mean_cdr2_A = np.mean(b_factors_cdr2_A)
    mean_cdr3_A = np.mean(b_factors_cdr3_A)
    mean_cdr1_B = np.mean(b_factors_cdr1_B)
    mean_cdr2_B = np.mean(b_factors_cdr2_B)
    mean_cdr3_B = np.mean(b_factors_cdr3_B)

    # Return the B-factors for each CDR region separately
    return mean_cdr1_A, mean_cdr2_A, mean_cdr3_A, mean_cdr1_B, mean_cdr2_B, mean_cdr3_B

def calculate_iptms(json_file_path, length=5):
    """
    Calculates the mean of `chain_iptm` and the mean of interface TCR-pMHC iPTMs
    using fixed chain mappings.

    Args:
        json_file_path (str): Path to the JSON file.

    Returns:
        dict: A dictionary with the calculated means.
    """
    try:
        # Load the JSON data from the file
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        # Calculate the mean of chain_iptm
        chain_iptm = data.get('chain_iptm', [])
        if not chain_iptm:
            print("No data found in 'chain_iptm'.")
            chain_iptm_mean = None
        else:
            chain_iptm_mean = sum(chain_iptm) / len(chain_iptm)
        
        # Calculate the mean for interface TCR-pMHC
        chain_pair_iptm = data.get('chain_pair_iptm', [])
        if not chain_pair_iptm:
            print("No data found in 'chain_pair_iptm'.")
            tcr_pmch_iptm = None
        else:
            # Fixed indices for TCR-pMHC interactions
            # A (MHC) = 0, B (B2M) = 1, C (peptide) = 2, D (TCRa) = 3, E (TCRb) = 4
            if length == 5:
                tcr_pmch_pairs = [
                    chain_pair_iptm[0][3],  # MHC-TCRa 
                    chain_pair_iptm[0][4],  # MHC-TCRb 
                    chain_pair_iptm[2][3],  # pep-TCRa
                    chain_pair_iptm[2][4]]  # pep-TCRb 
                tcr_pmch_iptm = sum(tcr_pmch_pairs) / len(tcr_pmch_pairs)
            elif length == 4:
                # A (MHC) = 0, C (peptide) = 1, D (TCRa) = 2, E (TCRb) = 3
                tcr_pmch_pairs = [
                    chain_pair_iptm[0][2],  # MHC-TCRa 
                    chain_pair_iptm[0][3],  # MHC-TCRb 
                    chain_pair_iptm[1][2],  # pep-TCRa
                    chain_pair_iptm[1][3]]  # pep-TCRb 
                tcr_pmch_iptm = sum(tcr_pmch_pairs) / len(tcr_pmch_pairs)
        return chain_iptm_mean, tcr_pmch_iptm
    
    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
        return None
    except KeyError as e:
        print(f"Missing key in JSON data: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def calculate_pdockq (model_file):
    pdockq_path = "../../data_augmentation/pdockq.py" #"/gpfs/projects/bsc72/aascunce/data_augmentation/pdockq.py"
    command=f"python {pdockq_path} --pdbfile {model_file}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    # Output is displayed as pDockQ = 0.609 for ./pre/merged_models_AB/1ao7_0_merged.pdb This corresponds to a PPV of at least 0.9400192
    # Capture pDockq
    pdockq = float(result.stdout.split('=')[1].split(' ')[1])
    return result.stdout, pdockq

def calculate_pdockq2_json (model_file, pae_mtx):
    pdockq2_path= "../../data_augmentation/pdockq2_pae.py" #"/gpfs/projects/bsc72/aascunce/data_augmentation/pdockq2_pae.py"
    command=f"python {pdockq2_path} -pae {pae_mtx} -pdb {model_file}"
    result=subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    ipae_A = float(result.stdout.split('\n')[1].split(' ')[1])
    ipae_B = float(result.stdout.split('\n')[2].split(' ')[1])
    pdockq2_A= float(result.stdout.split('\n')[4].split(' ')[1])
    pdockq2_B= float(result.stdout.split('\n')[5].split(' ')[1])
    return result.stdout, ipae_A, ipae_B, pdockq2_A, pdockq2_B

def extract_residues_and_resids(pdb_file, chain_id):
    """
    Extract the residue IDs and residues (in one-letter code) from a specific chain in a PDB file.
    
    Args:
        pdb_file (str): Path to the PDB file.
        chain_id (str): Chain ID to extract residues from.
    
    Returns:
        list of tuples: List of tuples where each tuple contains (resid, residue_one_letter).
    """
    
    parser=PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
        
    residues = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    resid = residue.get_id()[1]
                    resname = residue.get_resname()
                    residue_one_letter = PDB.Polypeptide.protein_letters_3to1.get(resname, 'X')  # Use 'X' for unknown residues
                    residues.append((resid, residue_one_letter))
    
    return residues

def run_anarci(sequence):
    """
    Execute ANARCI to assign IMGT numbering to a TCR sequence.
    """
    try:
        command=f"ANARCI -i {sequence} --scheme imgt"
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def parse_anarci_output(anarci_output):
    """
    Parse the output of ANARCI to extract IMGT numbering and residues, ensuring uniqueness of IMGT numbers.
    
    Args:
        anarci_output (str): Output from ANARCI as a string.
    
    Returns:
        list of tuples: A list where each tuple contains (IMGT_number, residue) with unique IMGT numbers.
    """
    pattern = r'^([A-Z])\s+(\d+)\s+([A-Z\-])'
    matches = re.findall(pattern, anarci_output, re.MULTILINE)
    
    imgt_numbered_seq = []
    seen_imgt_numbers = set()
    
    for match in matches:
        try:
            chain_letter, imgt_num, residue = match
            imgt_num = int(imgt_num)
            
            # Ensure uniqueness of IMGT numbers
            if imgt_num not in seen_imgt_numbers:
                imgt_numbered_seq.append((imgt_num, residue))
                seen_imgt_numbers.add(imgt_num)
        except ValueError as e:
            print(f"Error processing match: {match}. Error: {e}")
    
    return imgt_numbered_seq

def map_imgt_to_original(imgt_numbered_seq, pdb_resids):
    """
    Map the original numbering of a sequence from the PDB 'resids' to the IMGT numbering.
    
    Args:
        imgt_numbered_seq (list of tuples): The IMGT numbered sequence as tuples (IMGT_number, residue).
        pdb_resids (list of tuples): The original residue numbers from the PDB file as tuples (resid, residue_one_letter).
    
    Returns:
        list of tuples: A list where each tuple contains (original_resid, IMGT_number, residue).
    """
    mapping = []
    pdb_resid_index = 0  # Index for PDB residues
    
    for imgt_pos, residue in imgt_numbered_seq:
        if residue != "-":  # Only process non-gap residues in IMGT
            for original_resid, residue1 in pdb_resids[pdb_resid_index:]:
                if residue1 == residue:
                    mapping.append((original_resid, imgt_pos, residue))
                    pdb_resid_index += 1
                    break
                else:
                    pdb_resid_index += 1
            else:
                mapping.append((None, imgt_pos, residue))
        else:
            mapping.append((None, imgt_pos, residue))
    return mapping

def parse_CDR3 (mapping):
    cdr3_tuples = [tupple for tupple in mapping if 104 <= tupple[1] <= 118 and tupple[2] != "-"]
    return cdr3_tuples

def parse_CDR2 (mapping):
    cdr2_tuples = [tupple for tupple in mapping if 56 <= tupple[1] <= 65 and tupple[2] != "-"]
    return cdr2_tuples

def parse_CDR1 (mapping):
    cdr1_tuples = [tupple for tupple in mapping if 27 <= tupple[1] <= 38 and tupple[2] != "-"]
    return cdr1_tuples

def extract_atoms_for_cdr(cdr_list, pdb_file, chain_id):
    """
    Extract atoms of the residues of a CDR from a PDB file.
    
    :param cdr_list: List of tuples in the format (resid, imgtid, resname)
    :param pdb_file: Path to the PDB file of the structure
    :param chain_id: Chain ID (e.g., 'A', 'B') to extract the atoms from
    :return: List of tuples with the format (atomname, resid, resname, chain_id)
    """
    if pdb_file.endswith(".pdb"):
        parser = PDB.PDBParser(QUIET=True)
    else:
        parser=PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    atom_list = [] 
    for model in structure:
        for chain in model:
            if chain.id == chain_id:  
                for residue in chain:
                    resid = residue.get_id()[1]  
                    resname_3 = residue.get_resname() 
                    resname_1 = residue_mapping.get(resname_3, 'X')
                    for cdr_resid, cdr_imgtid, cdr_resname in cdr_list:
                        if resid == cdr_resid and resname_1 == cdr_resname:
                            for atom in residue:
                                atom_list.append((atom.get_name(), resid, resname_3, chain.id))
    return atom_list

def select_best_model(df):
    """
    Selects the best model for each TCR ID (PDB ID) based on the given priority of metrics:
    pDockQA > pDockQ > iptmTCR-MHC > iptm_mean > pLDDTCDR3b > pLDDTCDR3a > cdr pLDDTs > global pLDDT.

    Discards models that do not meet the following thresholds:
    - pDockQ2A < 0.23
    - Global or CDR pLDDTs < 70
    - pDockQ < 0.23
    - ipTM or TCRpmhc iptm < 0.60
    """

    # Step 1: Apply filtering to remove models below the thresholds
    filtered_df = df[
        (df['pdockq2_a'] >= 0.23) &
        (df['pdockq'] >= 0.23) &
        (df['iptm_tcrpmhc'] >= 0.60) &
        (df['iptm_mean'] >= 0.60) &
        (df[['cdr1a_plddt', 'cdr1b_plddt', 'cdr2a_plddt', 'cdr2b_plddt', 'cdr3a_plddt', 'cdr3b_plddt', 'plddt']].min(axis=1) >= 70)
    ]

    # Step 2: Sort the filtered DataFrame by the specified metrics in descending order
    sorted_df = filtered_df.sort_values(
        by=[
            'pdockq2_a',  # Combined pDockQA is most important
            'pdockq',     # Secondary importance
            'iptm_tcrpmhc',  # TCR-MHC interface ipTM
            'iptm_mean',     # Global ipTM mean
            'cdr3b_plddt',   # pLDDT of CDR3b
            'cdr3a_plddt',   # pLDDT of CDR3a
            'cdr1a_plddt', 'cdr1b_plddt', 'cdr2a_plddt', 'cdr2b_plddt',  # Other CDRs
            'plddt'          # Overall pLDDT
        ],
        ascending=False
    )
    
    # Step 3: Group by tcr_id (or pdb_id) and select the top model for each group
    best_models = sorted_df.groupby('tcr_id').first().reset_index()

    return best_models

def process_tcr_folders(base_path, threshold=70):
    """
    Processes TCR folders to clean CIF files and update PAE matrices, and writes metrics to a CSV file in each tcr folder.

    :param base_path: Path to the main TCR directory.
    :param threshold: PLDDT threshold for filtering residues.
    """
    
    tcr_id = base_path.split("_")[-1].rstrip("/")  # Extract tcr_id from folder name
    folder_path = base_path
    rows = []
     
    # Create the CSV file path for this specific tcr_id
    output_csv = os.path.join(folder_path, f"metrics_{tcr_id}.csv")
    file_exists = os.path.isfile(output_csv)
    
    # Open the CSV file for writing, append if exists
    with open(output_csv, mode='a', newline='') as csvfile:
        fieldnames = ['tcr_id', 'model', 'plddt', 'cdr1a_plddt', 'cdr1b_plddt', 'cdr2a_plddt', 'cdr2b_plddt',
                        'cdr3a_plddt', 'cdr3b_plddt', 'iptm_mean', 'iptm_tcrpmhc', 'pdockq', 'avg_ipae_a',
                        'avg_ipae_b', 'pdockq2_a', 'pdockq2_b']
                
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
        if not file_exists:
            writer.writeheader()  # Write header only if file does not exist

        for folder2 in os.listdir(folder_path):
            if folder2.startswith("seed"):
                # Extract model number from the seed folder name
                model_number = folder2.split("-")[-1]
                print(f"\nProcessing model {model_number} in TCR {tcr_id}...\n")
                folder2_path = os.path.join(folder_path, folder2)
                model_file_path = None
                file_output_path = None  
                summary_json_file_path = None
                confidence_json_file_path = None
                merged_pdb = None

                # Identify files in the seed folder
                for file in os.listdir(folder2_path):
                    if file.endswith(".cif"):
                        model_file_path = os.path.join(folder2_path, file)
                        file_output_path = os.path.join(folder_path, f"model_{model_number}.cif")
                                
                        # Remove low PLDDT residues and get removed atom indices
                        removed_atom_numbers = remove_low_plddt_and_get_absolute_indices(
                                    model_file_path, file_output_path, threshold=threshold)
                        sorted_removed_atom_numbers = sorted(removed_atom_numbers)
                        cif_to_pdb(file_output_path)
                        pdb_file_path = os.path.join(folder_path, f"model_{model_number}.pdb")
                        merge_pdb(pdb_file_path)
                        merged_pdb = os.path.join(folder_path, f"model_{model_number}_merged.pdb")

                    elif file.startswith("summary"):
                        summary_json_file_path = os.path.join(folder2_path, file)
                        print(f"Summary JSON file: {summary_json_file_path}")

                    elif file.startswith("confidences"):
                        confidence_json_file_path = os.path.join(folder2_path, file)
                        print(f"Confidence JSON file: {confidence_json_file_path}")
                        
                if confidence_json_file_path and sorted_removed_atom_numbers and merged_pdb:
                    # Update PAE matrix
                    pae_output_path = os.path.join(folder_path, f"pae_{model_number}.npy")
                    pae_matrix = remove_atoms_from_pae_matrix(confidence_json_file_path, sorted_removed_atom_numbers, pae_output_path)
                            
                    # Calculate all metrics
                    global_plddt = calculate_global_plddt(file_output_path)
                    cdr1a_plddt, cdr2a_plddt, cdr3a_plddt, cdr1b_plddt, cdr2b_plddt, cdr3b_plddt = cdr_plddts(file_output_path, "D", "E")
                    iptm_mean, iptm_tcrpmhc = calculate_iptms(summary_json_file_path)
                    _, pdockq = calculate_pdockq(merged_pdb)
                    _, avgipae_A, avgipae_B, pdockq2_A, pdockq2_B = calculate_pdockq2_json(merged_pdb, pae_output_path)
                    print(f"\nMetrics calculated for model {model_number} ---------------------------------------- ")        
                    print(f"Global PLDDT: {global_plddt}")
                    print(f"CDR1s PLDDT: CDR1a {cdr1a_plddt}, CDR1b {cdr1b_plddt}, CDR2a {cdr2a_plddt}, CDR2b {cdr2b_plddt}, CDR3a {cdr3a_plddt}, CDR3b {cdr3b_plddt}")
                    print(f"Chain IPTM mean: {iptm_mean}")
                    print(f"Interface TCR-pMHC IPTM mean: {iptm_tcrpmhc}")
                    print(f"pDockQ: {pdockq}")
                    print(f"Average iPAE A: {avgipae_A}, Average iPAE B: {avgipae_B}")
                    print(f"pDockQ2 A: {pdockq2_A}, pDockQ2 B: {pdockq2_B}")
                    print("------------------------------------------------------------------------")
                    # Append row to list 
                    rows.append({
                        'tcr_id': tcr_id,
                        'model': model_number,
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
                        'avg_ipae_a': avgipae_A,
                        'avg_ipae_b': avgipae_B,
                        'pdockq2_a': pdockq2_A,
                        'pdockq2_b': pdockq2_B})

                    # Write metrics to CSV
                    writer.writerow({
                                'tcr_id': tcr_id,
                                'model': model_number,
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
                                'avg_ipae_a': avgipae_A,
                                'avg_ipae_b': avgipae_B,
                                'pdockq2_a': pdockq2_A,
                                'pdockq2_b': pdockq2_B
                            })

                    # Remove files
                    os.remove(file_output_path)
                    os.remove(merged_pdb)

    metrics_df = pd.DataFrame(rows)
    return metrics_df

def main():
    # Add args with argpars
    parser = argparse.ArgumentParser(description="Process TCR folders to clean CIF files and update PAE matrices.")
    parser.add_argument("base_path", type=str, help="Path to the main TCR directory.")
    parser.add_argument("--threshold", type=int, default=70, help="PLDDT threshold for filtering residues.")
    args = parser.parse_args()

    print(f"Processing TCR folder... {args.base_path}")
    # Process TCR folders
    metrics_df = process_tcr_folders(args.base_path, threshold=args.threshold)
    best_model = select_best_model(metrics_df)
    if best_model.empty:
        print("No models passed the selection criteria.")
    else: 
        model_number = best_model['model'].iloc[0]
        tcr_id = best_model['tcr_id'].iloc[0]
        best_model_path = os.path.join(args.base_path, f"model_{model_number}.pdb")
        print(f"The best model for TCR {tcr_id} is model {model_number}.")
        best_model_folder =  "./best_models"
        os.makedirs(best_model_folder, exist_ok=True)
        cp_command = f"cp {best_model_path} {best_model_folder}/{tcr_id}_{model_number}.pdb"
        subprocess.run(cp_command, shell=True, check=True)
        print(f"Best model saved to {best_model_folder}/{tcr_id}_{model_number}.pdb")
        best_models_csv = "./best_models.csv"
        best_model_data = best_model
        if not os.path.isfile(best_models_csv):
            best_models_df = pd.DataFrame([best_model_data])
            best_models_df.to_csv(best_models_csv, index=False)
        else:
            best_models_df = pd.DataFrame([best_model_data])
            best_models_df.to_csv(best_models_csv, mode='a', header=False, index=False)
            
    print("Processing complete.")

if __name__ == "__main__":
    main()
