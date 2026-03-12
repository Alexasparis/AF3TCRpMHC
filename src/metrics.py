import os
import subprocess
import json
import numpy as np
from statistics import mean
from Bio.PDB import MMCIFParser, MMCIFIO, PDBParser
from anarci_utils import run_anarci, parse_anarci_output, map_imgt_to_original, parse_CDR1, parse_CDR2, parse_CDR3, extract_atoms_for_cdr
from utils import extract_sequences, extract_residues_and_resids, get_seq_from_pdb_chain

def remove_low_plddt_and_get_absolute_indices(cif_file, output_file, threshold=50):
    """
    Removes residues from the N- and C-termini of each chain in a CIF file if any atom in those residues has a B-factor (PLDDT) below the specified threshold.
    The function returns a list of absolute indices of the removed residues, which can be used to update the PAE and PDE matrices accordingly.
    :param cif_file: Path to the input CIF file
    :param output_file: Path to save the cleaned CIF file
    :param threshold: PLDDT threshold for trimming residues (default is 50)
    :return: List of absolute indices of removed residues
    :saves the cleaned CIF file as output_file
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_file)
    chain_lengths = {chain.id: len(list(chain.get_residues())) for chain in structure[0]}

    # Dictionary to track removed residue indices for each chain
    removed_indices = {chain.id: {"N": [], "C": []} for chain in structure[0]}

    model = structure[0]
    for chain in model:
        if chain.id=="C":
            continue
        residues = list(chain.get_residues())
        chain_id = chain.id
        
        # N-terminal trimming
        while residues:
            first_residue = residues[0]
            plddts = [atom.bfactor for atom in first_residue.get_atoms()]
            if any(plddt < threshold for plddt in plddts): 
                removed_indices[chain_id]["N"].append(first_residue.id[1])  
                chain.detach_child(first_residue.id)  
                residues.pop(0) 
            else:
                break
        
        # C-terminal trimming
        while residues:
            last_residue = residues[-1]
            plddts = [atom.bfactor for atom in last_residue.get_atoms()]
            if any(plddt < threshold for plddt in plddts):  
                removed_indices[chain_id]["C"].append(last_residue.id[1]) 
                chain.detach_child(last_residue.id) 
                residues.pop() 
            else:
                break

    # Save the cleaned CIF file
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(output_file)

    # List to hold absolute indices of removed residues
    absolute_indices = []
    global_residue_number = 1 

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
    Removes rows and columns corresponding to the removed residues from the PAE matrix in the JSON file and saves the updated matrix as a .npy file.
    :param json_file: Path to the original JSON file containing the PAE matrix
    :param removed_atom_numbers: List of absolute indices of removed residues
    :param output_file: Path to save the updated PAE matrix as a .npy file
    :return: Updated PAE matrix as a NumPy array
    :saves the updated PAE matrix as a .npy file at output_file
    """
    # Load the PAE matrix from the original JSON file
    with open(json_file, 'r') as f:
        pae_data = json.load(f)
    pae_matrix = np.array(pae_data['pae'])
    if removed_atom_numbers:
        pae_matrix = np.delete(pae_matrix, removed_atom_numbers, axis=0)
        pae_matrix = np.delete(pae_matrix, removed_atom_numbers, axis=1)
    np.save(output_file, pae_matrix)
    return pae_matrix

def remove_atoms_from_pde_matrix(json_file, removed_atom_numbers, output_file):
    """
    Removes rows and columns corresponding to the removed residues from the PDE matrix in the JSON file and saves the updated matrix as a .npy file.
    :param json_file: Path to the original JSON file containing the PDE matrix
    :param removed_atom_numbers: List of absolute indices of removed residues
    :param output_file: Path to save the updated PDE matrix as a .npy file
    :return: Updated PDE matrix as a NumPy array
    :saves the updated PDE matrix as a .npy file at output_file
    """
    # Load the conntact_probs matrix from the original JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    pde_matrix = np.array(data['contact_probs'])
    if removed_atom_numbers:
        pde_matrix = np.delete(pde_matrix, removed_atom_numbers, axis=0)
        pde_matrix = np.delete(pde_matrix, removed_atom_numbers, axis=1)
    np.save(output_file, pde_matrix)
    return pde_matrix

def calculate_global_plddt(cif_file_path):
    """
    Calculates the global pLDDT for a CIF file.
    :param cif_file_path: Path to the input CIF file
    :return: Global pLDDT value 
    """

    try:
        # Parse the CIF file
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("structure", cif_file_path)

        # Extract all B-factors
        b_factors = [atom.get_bfactor() for model in structure for chain in model for residue in chain for atom in residue]

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
    Extracts B-factors for the specified CDR atoms from the given chain.
    :param cdr_atoms: List of tuples containing (atomname, resid, resname, chainid) for CDR atoms
    :param chain: Biopython Chain object from which to extract B-factors
    :return: List of B-factors corresponding to the CDR atoms
    """
    b_factors = []
    for atomname, resid, resname, chainid in cdr_atoms:
        if chainid == chain.id:
            try:
                residue = chain[resid] 
                if atomname in residue:  
                    atom = residue[atomname]
                    b_factors.append(atom.get_bfactor())
                else:
                    print(f"Atom {atomname} not found in residue {resid} ({resname}) of chain {chain.id}")
            except KeyError:
                print(f"Residue {resid} ({resname}) not found in chain {chain.id}")
    return b_factors

def cdr_plddts(model_file, alpha_chain="D", beta_chain="E"):
    """
    Calculate mean B-factors (PLDDT) for CDR regions of alpha and beta chains.

    :param model_file: Path to PDB or CIF model file
    :param alpha_chain: Chain ID for alpha chain
    :param beta_chain: Chain ID for beta chain
    :return: mean_cdr1_A, mean_cdr2_A, mean_cdr3_A, mean_cdr1_B, mean_cdr2_B, mean_cdr3_B
    """

    # --- Extract sequences and residues ---
    model_sequences, model_dict = extract_sequences(model_file)
    residues_A = extract_residues_and_resids(model_file, alpha_chain)
    residues_B = extract_residues_and_resids(model_file, beta_chain)

    # --- Run ANARCI and parse CDR regions ---
    parsed_A = parse_anarci_output(run_anarci(model_sequences[alpha_chain]))
    parsed_B = parse_anarci_output(run_anarci(model_sequences[beta_chain]))

    map_A = map_imgt_to_original(parsed_A, residues_A)
    map_B = map_imgt_to_original(parsed_B, residues_B)

    # Extract CDR residues
    cdr1_A, cdr2_A, cdr3_A = parse_CDR1(map_A), parse_CDR2(map_A), parse_CDR3(map_A)
    cdr1_B, cdr2_B, cdr3_B = parse_CDR1(map_B), parse_CDR2(map_B), parse_CDR3(map_B)

    # Extract atom info for each CDR
    cdr_atoms = {
        "cdr1_A": extract_atoms_for_cdr(cdr1_A, model_file, alpha_chain),
        "cdr2_A": extract_atoms_for_cdr(cdr2_A, model_file, alpha_chain),
        "cdr3_A": extract_atoms_for_cdr(cdr3_A, model_file, alpha_chain),
        "cdr1_B": extract_atoms_for_cdr(cdr1_B, model_file, beta_chain),
        "cdr2_B": extract_atoms_for_cdr(cdr2_B, model_file, beta_chain),
        "cdr3_B": extract_atoms_for_cdr(cdr3_B, model_file, beta_chain),
    }

    # --- Parse structure ---
    parser = PDBParser(QUIET=True) if model_file.endswith(".pdb") else MMCIFParser(QUIET=True)
    structure = parser.get_structure("Model", model_file)
    chain_A = structure[0][alpha_chain]
    chain_B = structure[0][beta_chain]

    # --- Helper to calculate mean B-factor safely ---
    def mean_b_factors(atoms, chain, cdr_name):
        if not atoms:
            print(f"ERROR: {cdr_name} empty in model {model_file}")
            return np.nan
        return np.mean(extract_b_factors(atoms, chain))

    # --- Compute mean B-factors for all CDRs ---
    mean_cdr1_A = mean_b_factors(cdr_atoms["cdr1_A"], chain_A, "CDR1_A")
    mean_cdr2_A = mean_b_factors(cdr_atoms["cdr2_A"], chain_A, "CDR2_A")
    mean_cdr3_A = mean_b_factors(cdr_atoms["cdr3_A"], chain_A, "CDR3_A")

    mean_cdr1_B = mean_b_factors(cdr_atoms["cdr1_B"], chain_B, "CDR1_B")
    mean_cdr2_B = mean_b_factors(cdr_atoms["cdr2_B"], chain_B, "CDR2_B")
    mean_cdr3_B = mean_b_factors(cdr_atoms["cdr3_B"], chain_B, "CDR3_B")

    return mean_cdr1_A, mean_cdr2_A, mean_cdr3_A, mean_cdr1_B, mean_cdr2_B, mean_cdr3_B

def calculate_iptms(json_file_path, length=5):
    """
    Calculates the mean of `chain_iptm` and the mean of interface TCR-pMHC iPTMs
    using fixed chain mappings.
    :param json_file_path: Path to the JSON file containing the iPTM data
    :param length: Number of chains in the model (4 or 5) to determine chain mappings
    :return: A tuple containing the mean of `chain_iptm` and the mean of interface TCR-pMHC iPTMs
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
            # A (MHC) = 0, B (B2M) = 1, C (peptide) = 2, D (TCRA) = 3, E (TCRB) = 4
            if length == 5:
                tcr_pmch_pairs = [
                    chain_pair_iptm[0][3],  # MHC-TCRa 
                    chain_pair_iptm[0][4],  # MHC-TCRb 
                    chain_pair_iptm[2][3],  # pep-TCRa
                    chain_pair_iptm[2][4]]  # pep-TCRb 
                tcr_pmch_iptm = sum(tcr_pmch_pairs) / len(tcr_pmch_pairs)
            elif length == 4:
                # A (TCRA) = 0, C (peptide) = 1, D (TCRa) = 2, E (TCRb) = 3
                tcr_pmch_pairs = [
                    chain_pair_iptm[0][2],  # MHC-TCRa 
                    chain_pair_iptm[0][3],  # MHC-TCRb 
                    chain_pair_iptm[2][2],  # pep-TCRa
                    chain_pair_iptm[2][3]]  # pep-TCRb 
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

def calculate_pdockq (model_file, scripts_path="./"):
    """
    Calculates the pDockQ score for a given PDB model file using the pdockq.py script.
    :param model_file: Path to the input PDB file
    :param script_path: Path to the pdockq.py script
    :return: A tuple containing the raw output from pdockq.py and the extracted pDockQ score as a float
    """
    pdockq_path = scripts_path + "pdockq.py"
    command=f"python {pdockq_path} --pdbfile {model_file}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    # Output is displayed as pDockQ = 0.609 for ./pre/merged_models_AB/1ao7_0_merged.pdb This corresponds to a PPV of at least 0.9400192
    # Capture pDockq
    pdockq = float(result.stdout.split('=')[1].split(' ')[1])
    return result.stdout, pdockq

def calculate_pdockq2(model_file, pde_mtx, pae_mtx, scripts_path="./"):
    """
    Calculates the pDockQ2 score for a given PDB model file using the pdockq2.py script, which takes into account both the PDE and PAE matrices.
    :param model_file: Path to the input PDB file
    :param pde_mtx: Path to the PDE matrix file (in .npy format)
    :param pae_mtx: Path to the PAE matrix file (in .npy format)
    :param script_path: Path to the pdockq2.py script
    :return: A tuple containing the raw output from pdockq2.py and the extracted scores for IPDE, IPAE, and pDockQ2 for both chains A and B as floats
    """
    pdockq2_path = scripts_path + "pdockq2.py"
    command = f'"python" "{pdockq2_path}" -pde "{pde_mtx}" -pae "{pae_mtx}" -pdb "{model_file}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error")
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        return None, None, None, None, None, None, None
    lines = [line for line in result.stdout.strip().split('\n') if line.strip()]

    try:
        ipde_A = float(lines[1].split()[1])
        ipde_B = float(lines[2].split()[1])
        ipae_A = float(lines[4].split()[1])
        ipae_B = float(lines[5].split()[1])
        pdockq2_A = float(lines[7].split()[1])
        pdockq2_B = float(lines[8].split()[1])
    except (IndexError, ValueError) as e:
        print("Error parseando la salida de pdockq2.py:", e)
        print("Salida completa:\n", result.stdout)
        return None, None, None, None, None

    return result.stdout, ipde_A, ipde_B, ipae_A, ipae_B, pdockq2_A, pdockq2_B


def get_sync_indices_for_pae(json_path, pdb_path):
    """
    Determines the absolute indices of residues that are present in the original sequence but missing in the PDB structure, which should be removed from the PAE matrix.
    :param json_path: Path to the JSON file containing the original sequences
    :param pdb_path: Path to the PDB file containing the actual sequences of the chains
    :return: A list of absolute indices of residues that should be removed from the PAE matrix to synchronize it with the actual sequences in the PDB structure
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    original_seqs = {}
    for item in data.get('sequences', []):
        if 'protein' in item:
            original_seqs[item['protein']['id']] = item['protein']['sequence']
    
    ordered_chains = sorted(original_seqs.keys())
    
    absolute_removed_indices = []
    current_pae_offset = 0
    for chain_id in ordered_chains:
        full_seq = original_seqs[chain_id]
        len_full = len(full_seq)
        pdb_seq = get_seq_from_pdb_chain(pdb_path, chain_id)
        #print(f"Chain {chain_id}:")
        #print(f"Full length {len_full}", full_seq)
        #print(f"PDB length {len(pdb_seq)}", pdb_seq)
        
        if not pdb_seq:
            # If the whole chain is missing from PDB, remove all its indices from PAE
            print(f"Chain {chain_id} missing in PDB, removing all its residues from PAE.")
            for i in range(len_full):
                absolute_removed_indices.append(current_pae_offset + i)
        else:
            start_rel = full_seq.find(pdb_seq)
            
            if start_rel != -1:
                # Add indices for residues removed at the Start (N-term)
                for i in range(0, start_rel):
                    absolute_removed_indices.append(current_pae_offset + i)
                #print(f"Residues removed at start (N-term): {start_rel}")
                # Add indices for residues removed at the End (C-term)
                end_rel = start_rel + len(pdb_seq)
                for i in range(end_rel, len_full):
                    absolute_removed_indices.append(current_pae_offset + i)
                #print(f"Residues removed at end (C-term): {len_full - end_rel}")
            else:
                print(f"Warning: Sequence mismatch in chain {chain_id}")

        # Move the offset by the FULL original length of the chain
        current_pae_offset += len_full
        
    #print(f"Total removed residues from PAE: {len(absolute_removed_indices)}")
    return absolute_removed_indices

def has_tcr_peptide_contact(structure, cutoff=10.0):
    """
        Determines if there is a contact between TCR and peptide chains in the structure based on a distance cutoff.
        :param structure: Biopython Structure object representing the protein complex
        :param cutoff: Distance cutoff in angstroms to define a contact between TCR and peptide atoms
        :return: True if a contact is detected between TCR and peptide chains, False otherwise
    """
    tcr_atoms = {'D': [], 'E': []}
    pep_atoms = []

    model = structure[0]

    # collect atom coordinates
    for chain in model:
        cid = chain.id

        # peptide chain
        if cid == 'C':
            for residue in chain:
                for atom in residue:
                    pep_atoms.append(atom.get_coord())

        # TCR alpha (A) and beta (B)
        if cid in ('D', 'E'):
            for residue in chain:
                for atom in residue:
                    tcr_atoms[cid].append(atom.get_coord())

    # if no peptide, no contact is possible
    if not pep_atoms:
        return False
    pep_arr = np.array(pep_atoms)

    # check A and B separately
    for cid in ('D', 'E'):
        if not tcr_atoms[cid]:
            continue

        tcr_arr = np.array(tcr_atoms[cid])
        diffs = tcr_arr[:, None, :] - pep_arr[None, :, :]
        dists = np.sqrt(np.sum(diffs * diffs, axis=2))

        if np.min(dists) <= cutoff:
            return True
    return False

def calculate_ipsae (cif_file_path, json_conf_path, scripts_dir):
    """
    Calculates the iPSAE score for a given CIF file using the ipsae.py script.
    :param cif_file_path: Path to the input CIF file
    :param json_conf_path: Path to the JSON configuration file required by ipsae.py
    :param scripts_dir: Directory where the ipsae.py script is located
    :saves the iPSAE results
    """
    command = ["python", os.path.join(scripts_dir, "ipsae.py"), json_conf_path, cif_file_path, "10", "10"]
    print(f"Calculating iPSAE for {cif_file_path}...")
    print(" ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error calculating iPSAE for {cif_file_path}: {result.stderr}")

def parse_ipsae_output(output_file_path):
    """
    Parses the output file generated by ipsae.py to extract relevant metrics.
    :param output_file_path: Path to the output file generated by ipsae.py
    :return: A dictionary containing the extracted metrics for each interface and type (max, mean, min) with keys formatted as 
    "Chn1_Chn2_Type" and values as dictionaries of the metrics (ipSAE, ipSAE_d0chn, ipSAE_d0dom, ipTM_d0chn, LIS)    
    """
    results = {}

    try:
        with open(output_file_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        header = lines[0].split()

        target_cols = [
            "ipSAE",
            "ipSAE_d0chn",
            "ipSAE_d0dom",
            "ipTM_d0chn",
            "LIS"]

        col_idx = {col: header.index(col) for col in target_cols}

        # Parse all data rows (skip header)
        for line in lines[1:]:
            fields = line.split()

            chn1 = fields[header.index("Chn1")]
            chn2 = fields[header.index("Chn2")]
            row_type = fields[header.index("Type")]

            key = f"{chn1}_{chn2}_{row_type}"

            results[key] = {col: float(fields[idx]) for col, idx in col_idx.items()}

        return results

    except Exception as e:
        print(f"Error parsing ipSAE output file {output_file_path}: {e}")
        return {}


def calculate_ipsae_for_seed(folder_path, cif_file, confidence_file, scripts_dir):
    """
    Calculates the iPSAE score for a given seed by checking for existing results and running the calculation if necessary,
    then extracts the relevant metrics for each interface and computes mean values across interfaces.
    
    :param folder_path: Path to the folder containing the CIF file, confidence file, and potentially existing iPSAE results
    :param cif_file: Path to the CIF file for the current seed
    :param confidence_file: Path to the JSON confidence file for the current seed
    :param scripts_dir: Directory where the ipsae.py script is located
    :return: A tuple containing the mean iPSAE score, mean iPSAE_d0chn, mean iPSAE_d0dom, mean ipTM_d0chn, and mean LIS
             across the interfaces, or None if the calculation could not be performed due to missing files or errors.
    """

    # Search for existing iPSAE results in the folder
    ipsae_path = next(
        (os.path.join(folder_path, f) for f in os.listdir(folder_path)
         if f.endswith(".txt") and not f.endswith("byres.txt")),
        None
    )

    if ipsae_path is None:
        print(f"iPSAE results not found in {folder_path}. Running calculation...")
        if cif_file and confidence_file:
            calculate_ipsae(cif_file, confidence_file, scripts_dir=scripts_dir)
            # Try to find the result file again
            ipsae_path = next(
                (os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.endswith(".txt") and not f.endswith("byres.txt")),
                None
            )
        else:
            print(f"Error: Missing CIF or JSON file in {folder_path}")
            return None

    if ipsae_path is None:
        print(f"Error: iPSAE calculation failed in {folder_path}")
        return None

    # Parse the iPSAE results
    ipsae_results = parse_ipsae_output(ipsae_path)

    interfaces = ["A_D", "A_E", "C_D", "C_E"]
    results = {}

    for interface in interfaces:
        results[interface] = (
            ipsae_results.get(f"{interface}_max", {}).get("ipSAE", None),
            ipsae_results.get(f"{interface}_max", {}).get("ipSAE_d0chn", None),
            ipsae_results.get(f"{interface}_max", {}).get("ipSAE_d0dom", None),
            ipsae_results.get(f"{interface}_max", {}).get("ipTM_d0chn", None),
            ipsae_results.get(f"{interface}_max", {}).get("LIS", None),
        )

    # Compute mean values (ignore None values)
    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    ipsae = mean([results[iface][0] for iface in interfaces])
    ipsae_d0chn = mean([results[iface][1] for iface in interfaces])
    ipsae_d0dom = mean([results[iface][2] for iface in interfaces])
    iptm_d0chn = mean([results[iface][3] for iface in interfaces])
    lis = mean([results[iface][4] for iface in interfaces])

    return ipsae, ipsae_d0chn, ipsae_d0dom, iptm_d0chn, lis