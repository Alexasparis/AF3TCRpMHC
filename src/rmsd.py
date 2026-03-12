
import os
import numpy as np
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Superimposer import SVDSuperimposer
from Bio.Align import PairwiseAligner
import subprocess
import sys
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from utils import extract_sequences, extract_residues_and_resids
from anarci_utils import run_anarci, parse_anarci_output, map_imgt_to_original, parse_CDR3, extract_atoms_for_cdr

def align_sequences(seqA, seqB):
    """
    Align two sequences using pairwise sequence alignment and return the aligned sequences with residue info.
    """
    aligner = PairwiseAligner()
    aligner.match = 5
    aligner.mismatch = -1
    aligner.open_gap_score = -4
    aligner.extend_gap_score = -1
    aln = aligner.align(seqA, seqB)[0]
    return aln

def format_alignment(aln, chain_id_cry, chain_id_mod, dict_cry, dict_mode):
    """
    Format the alignment and return the residue names and IDs for each sequence in the alignment.
    Also return the matches as before ('|' for match, '.' for mismatch, ' ' for gap).
    """
    aligned_residues = {'seqA': [], 'seqB': [], 'matches': []}
    seqA_aligned = aln[0, :]
    seqB_aligned = aln[1, :]
    indexA = 0
    indexB = 0
    for i in range(len(seqA_aligned)):
        if seqA_aligned[i] != '-':
            res_nameA, res_idA = dict_cry[chain_id_cry][indexA] 
            aligned_residues['seqA'].append((res_nameA, res_idA))
            indexA += 1
        else:
            aligned_residues['seqA'].append(('-', '-'))

        if seqB_aligned[i] != '-':
            res_nameB, res_idB = dict_mode[chain_id_mod][indexB] 
            aligned_residues['seqB'].append((res_nameB, res_idB))
            indexB += 1
        else:
            aligned_residues['seqB'].append(('-', '-'))

        if seqA_aligned[i] == seqB_aligned[i]:
            aligned_residues['matches'].append('|')  
        elif seqA_aligned[i] == '-' or seqB_aligned[i] == '-':
            aligned_residues['matches'].append(' ')  
        else:
            aligned_residues['matches'].append('.')  
    return aligned_residues

def get_aligned_residues(alignment):
    """
    Map aligned residues from two chains based on a given sequence alignment.
    Only returns residues that are matches, formatted as (resname, resid_crystal, resid_model).
    """
    aligned_residues = []
    seqA_aligned = alignment["seqA"]
    seqB_aligned = alignment["seqB"]
    matches = alignment["matches"]
    indexA = 0
    indexB = 0
    for i in range(len(seqA_aligned)):
        if seqA_aligned[i] != "-" and seqB_aligned[i] != "-" and matches[i] == "|":
            res_nameA, resid_crystal = seqA_aligned[i]
            res_nameB, resid_model = seqB_aligned[i]
            aligned_residues.append((res_nameA, resid_crystal, resid_model))
            indexA += 1
            indexB += 1
    return aligned_residues

def get_interface(pdb_file, reference_chain, chain_ids, distance_cutoff=10.0, select_heavy_atoms=False):
    """
    Select Cα atoms from chains A, B, and C that are within a certain distance (default: 10 Å)
    from any atom of chain C, and also include all Cα atoms from chain C. Optionally, select heavy atoms.

    Args:
    - pdb_file (str): Path to the PDB file.
    - reference_chain (str): The chain ID of the reference chain (usually chain C).
    - chain_ids (list of str): List of chain IDs to select atoms from, including 'A', 'B', and 'C'.
    - distance_cutoff (float): Distance cutoff in Å (default 10 Å).
    - select_heavy_atoms (bool): If True, select all heavy atoms (default: False).

    Returns:
    - selected_atoms (list of tuples): A list of atoms (atom name, residue ID, residue name, and chain ID)
      that are within the distance cutoff from atoms in chain C, plus all Cα atoms from chain C (or all heavy atoms if specified).
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    selected_atoms = []
    chain_ref_atoms = []
    chain_others_atoms = []
    for model in structure:
        for chain in model.get_chains():
            chain_id = chain.get_id()
            if chain_id == reference_chain:
                for residue in chain:
                    for atom in residue:
                        chain_ref_atoms.append(atom)
            if chain_id in chain_ids:
                for residue in chain:
                    for atom in residue:
                        if select_heavy_atoms:
                            if atom.element != 'H':
                                chain_others_atoms.append((atom, residue.get_id()[1], residue.get_resname(), chain_id))
                        else:
                            if atom.get_name() == "CA":
                                chain_others_atoms.append((atom, residue.get_id()[1], residue.get_resname(), chain_id))
    for atom, resid, resname, chain_id in chain_others_atoms:
        for ref_atom in chain_ref_atoms:
            distance = atom - ref_atom
            if distance <= distance_cutoff:
                selected_atoms.append((atom.get_name(), resid, resname, chain_id))
    for atom in chain_others_atoms:
        atom_obj, resid, resname, chain_id = atom
        if chain_id == reference_chain:  # Only add atoms from chain C
            selected_atoms.append((atom_obj.get_name(), resid, resname, chain_id))
    return selected_atoms

def get_atom_coordinates(pdb_file, selected_atoms):
    if pdb_file.endswith(".pdb"):
        parser = PDB.PDBParser(QUIET=True)
    else:
        parser = PDB.MMCIFParser(QUIET=True)
    
    structure = parser.get_structure('structure', pdb_file)
    coordinates = []

    for atom_name, resid, resname, chain_id in selected_atoms:
        try:
            chain = structure[0][chain_id]
            residue = chain[resid]
            found = False
            for atom in residue:
                if atom.get_name() == atom_name:
                    coordinates.append(atom.get_coord())
                    found = True
                    break
            if not found:
                print(f"Warning: Atom {atom_name} in residue {resid} of chain {chain_id} not found.")
        except KeyError:
            print(f"Warning: Chain {chain_id} or residue {resid} not found in the structure.")

    if not coordinates:
        raise ValueError(f"No coordinates found for the selected atoms: {selected_atoms}")

    coordinates_array = np.array(coordinates, dtype='f')
    return coordinates_array


def calculate_rmsd(crystal_pdb, model_pdb, pdb_id, chain_dict, distance_cutoff=10.0, model_pairwise = False):
    """
    Calculate RMSD between a crystal and a model structure for all chains.

    Args:
        crystal_pdb (str): Path to the crystal structure PDB file.
        model_pdb (str): Path to the model structure PDB file.
        chain_ids (list): List of chain IDs to consider for the interface (default ['A', 'B', 'C']).
        distance_cutoff (float): Distance cutoff for selecting interface residues (default 10.0 Å).
        model_pairwise (bool): If True, calculate RMSD for each model pair in the same PDB ID.

    Returns:
        dict: Overall RMSD and chain-specific RMSD details.
    """
    
    # Step 1: Parse structures and extract sequence
    if model_pdb.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    else:
        parser = MMCIFParser(QUIET=True)
    model_structure = parser.get_structure("model", model_pdb)

    # Extract sequences and map residues
    crystal_sequences, dict_cry = extract_sequences(crystal_pdb)
    model_sequences, dict_mod = extract_sequences(model_pdb)
    if model_pairwise:  
        chain_dict={
            'mhc_chain':'A',
            'b2_chain':'B',
            'peptide_chain':'C',
            'tcra_chain':'D',
            'tcrb_chain':'E'}
    else:
        chain_dict=chain_dict[pdb_id]
    mapping = []

    if len(model_sequences) == 5:
        model_chain_mapping = {
            'mhc_chain':'A',
            'b2_chain':'B',
            'peptide_chain':'C',
            'tcra_chain':'D',
            'tcrb_chain':'E'}
        
    elif len(model_sequences) == 4:
        model_chain_mapping = {
            'mhc_chain':'A',
            'peptide_chain':'B',
            'tcra_chain':'C',
            'tcrb_chain':'D'}
    else:
        print(f"Error: Model structure does not have 4 or 5 chains.")

    # Step 2: Loop over chains to align sequences
    for chain_crystal, seq_crystal in crystal_sequences.items():
        key = next((key for key, value in chain_dict.items() if value == chain_crystal), None)
        if key is not None:
            model_id = model_chain_mapping.get(key)
            if model_id in model_sequences:
                seq_model = model_sequences[model_id]
                alignment = align_sequences(seq_crystal, seq_model)
                formatted_alignment = format_alignment(alignment, chain_crystal, model_id, dict_cry, dict_mod)
                aligned_residues = get_aligned_residues(formatted_alignment)
                aligned_residues = [res + (f'{chain_crystal}',) for res in aligned_residues]
                mapping.extend(aligned_residues)
            else:
                print(f"Error: {model_id} not in model_sequences.")
        else:
            print(f"Error: {chain_crystal} not in chain_dict.")
            print(f"Available chains in chain_dict: {chain_dict}")

    # Step 3: Select interface atoms
    reference_chain = chain_dict['peptide_chain']
    if len(model_sequences) == 5:
        chain_ids = [chain_dict['tcra_chain'], chain_dict['tcrb_chain'], chain_dict['peptide_chain'], chain_dict['mhc_chain'], chain_dict['b2_chain']]
    elif len(model_sequences) == 4:
        chain_ids = [chain_dict['tcra_chain'], chain_dict['tcrb_chain'], chain_dict['peptide_chain'], chain_dict['mhc_chain']]
 
    selected_atoms_crystal = sorted(set(get_interface(crystal_pdb, reference_chain, chain_ids=chain_ids, distance_cutoff=distance_cutoff, select_heavy_atoms=True)),key=lambda x: (x[3], x[1]))
    selected_atoms_model = []
    atoms_to_remove_i = []
    for atom_crystal in selected_atoms_crystal:
        atom_name_crystal, resid_crystal, resname_crystal, chain_id_crystal = atom_crystal
        found_match = False  
        for resname, resid_crystal_mapping, resid_model_mapping, chain_id_mapping in mapping:
            if resid_crystal == resid_crystal_mapping and resname_crystal == resname and chain_id_crystal == chain_id_mapping:
                chain_ident = next((key for key, value in chain_dict.items() if value == chain_id_mapping), None)
                chain_id_model = model_chain_mapping.get(chain_ident)
                chain_model = model_structure[0][chain_id_model]
                for residue_model in chain_model:
                    if residue_model.get_id()[1] == resid_model_mapping:
                        for atom_model in residue_model:
                            if atom_model.get_name() == atom_name_crystal:
                                selected_atoms_model.append((atom_model.get_name(), resid_model_mapping, resname, chain_id_model))
                                found_match = True  # Marcar como encontrado
                                break
                    if found_match:  
                        break
            if found_match:
                break
        if not found_match:
            atoms_to_remove_i.append(atom_crystal)
    selected_atoms_crystal = [atom for atom in selected_atoms_crystal if atom not in atoms_to_remove_i]

    # Step 4: Get atom coordinates
    coordinates_crystal = get_atom_coordinates(crystal_pdb, selected_atoms_crystal)
    coordinates_model = get_atom_coordinates(model_pdb, selected_atoms_model)

    # Step5: Get CDR3 residues and atoms
    residues_crystal_A=extract_residues_and_resids(crystal_pdb, chain_dict['tcra_chain'])
    residues_crystal_B=extract_residues_and_resids(crystal_pdb, chain_dict['tcrb_chain'])

    anarci_A_cry=run_anarci(crystal_sequences[chain_dict['tcra_chain']])
    anarci_B_cry=run_anarci(crystal_sequences[chain_dict['tcrb_chain']])
    
    parsed_cry_A=parse_anarci_output(anarci_A_cry)
    parsed_cry_B=parse_anarci_output(anarci_B_cry)

    map_cry_A=map_imgt_to_original(parsed_cry_A, residues_crystal_A)
    map_cry_B=map_imgt_to_original(parsed_cry_B, residues_crystal_B)
    
    cdr3_cry_A=parse_CDR3(map_cry_A)
    cdr3_cry_B=parse_CDR3(map_cry_B)
    
    cdr_atoms_cry_A = extract_atoms_for_cdr(cdr3_cry_A, crystal_pdb, chain_dict['tcra_chain'])
    cdr_atoms_cry_B = extract_atoms_for_cdr(cdr3_cry_B, crystal_pdb, chain_dict['tcrb_chain'])
    cdr_atoms_crystal = cdr_atoms_cry_A + cdr_atoms_cry_B

    cdr_atoms_model = []
    atoms_to_remove = []

    for cdr_atom_crystal in cdr_atoms_crystal:
        cdr_atom_name_crystal, cdr_resid_crystal, cdr_resname_crystal, cdr_chain_id_crystal = cdr_atom_crystal
        found_match = False 
        for cdr_resname, cdr_resid_crystal_mapping, cdr_resid_model_mapping, cdr_chain_id_mapping in mapping:
            if cdr_resid_crystal == cdr_resid_crystal_mapping and cdr_resname_crystal == cdr_resname and cdr_chain_id_crystal == cdr_chain_id_mapping:
                cdr_chain_ident = next((key for key, value in chain_dict.items() if value == cdr_chain_id_mapping), None)
                cdr_chain_id_model = model_chain_mapping.get(cdr_chain_ident)
                chain_model = model_structure[0][cdr_chain_id_model]
                for residue_model in chain_model:
                    if residue_model.get_id()[1] == cdr_resid_model_mapping:  # Match the residue ID
                        for atom_model in residue_model:
                            if atom_model.get_name() == cdr_atom_name_crystal:  # Match the atom name
                                cdr_atoms_model.append((atom_model.get_name(), cdr_resid_model_mapping, cdr_resname_crystal, cdr_chain_id_model))
                                found_match = True  # Mark as found
                                break
                        if found_match:  # Break outer loops if a match is found
                            break
                if found_match:
                    break
        if not found_match:
            atoms_to_remove.append(cdr_atom_crystal)

    cdr_atoms_crystal = [atom for atom in cdr_atoms_crystal if atom not in atoms_to_remove]

    # Perform superposition and calculate overall RMSD
    if len(coordinates_crystal) == len(coordinates_model) and len(coordinates_crystal) > 0:
        sup = SVDSuperimposer()
        sup.set(coordinates_crystal, coordinates_model)
        sup.run()
        y_on_x = sup.get_transformed()
        overall_rmsd = sup.get_rms()
    else:
        return "Error: Interface atoms mismatch. Cannot superimpose."

    # Step 6: Categorize chains and calculate RMSD
    if len(model_sequences) == 5:
        categories_crystal = {
            'TCRA/TCRB': [chain_dict['tcra_chain'], chain_dict['tcrb_chain']],
            'Peptide': [chain_dict['peptide_chain']],
            'MHC/B2M': [chain_dict['mhc_chain'], chain_dict['b2_chain']]}

        categories_model = {
            'TCRA/TCRB': [model_chain_mapping['tcra_chain'], model_chain_mapping['tcrb_chain']],
            'Peptide': [model_chain_mapping['peptide_chain']],
            'MHC/B2M': [model_chain_mapping['mhc_chain'], model_chain_mapping['b2_chain']]}
    
    elif len(model_sequences) == 4:
        categories_crystal = {
            'TCRA/TCRB': [chain_dict['tcra_chain'], chain_dict['tcrb_chain']],
            'Peptide': [chain_dict['peptide_chain']],
            'MHC': [chain_dict['mhc_chain']]}
        
        categories_model = {
            'TCRA/TCRB': [model_chain_mapping['tcra_chain'], model_chain_mapping['tcrb_chain']],
            'Peptide': [model_chain_mapping['peptide_chain']],
            'MHC': [model_chain_mapping['mhc_chain']]}
    else:
        return "Error: Model structure does not have 4 or 5 chains."

    category_crystal_coords = {category: [] for category in categories_crystal}
    category_model_coords = {category: [] for category in categories_model}
    
    for atom, coord in zip(selected_atoms_crystal, coordinates_crystal):
        chain_id = atom[3]
        for category, chains in categories_crystal.items():
            if chain_id in chains:
                category_crystal_coords[category].append(coord)
                break 

    for atom, coord in zip(selected_atoms_model, y_on_x):
        chain_id = atom[3]
        for category, chains in categories_model.items():
            if chain_id in chains:
                category_model_coords[category].append(coord)
                break  

    # Step 6: Calculate RMSD for each category
    category_rmsd_results = {}
    for category, crystal_coords in category_crystal_coords.items():
        model_coords = category_model_coords[category]
        
        # Check if coordinates are valid and have the same length
        if len(crystal_coords) > 0 and len(crystal_coords) == len(model_coords):
            # Convert to numpy arrays for easy manipulation
            crystal_coords = np.array(crystal_coords)
            model_coords = np.array(model_coords)
            
            # Calculate the difference between the coordinates
            diff = crystal_coords - model_coords
            
            # Calculate RMSD: sqrt(sum((crystal - model)^2) / N)
            rmsd = np.sqrt(np.sum(np.square(diff)) / len(crystal_coords))
            category_rmsd_results[category] = rmsd
        else:
            category_rmsd_results[category] = None

    # Prepare result string
    result_string = f"Number of CA in interface: {len(selected_atoms_crystal)}, Overall iRMSD: {overall_rmsd:.2f} angstroms\n"
    for category, rmsd in category_rmsd_results.items():
        num_atoms = len(category_crystal_coords[category])
        
        if rmsd is not None:
            result_string += f"{category}: Number of CA: {num_atoms}, iRMSD: {rmsd:.2f} angstroms\n"
        else:
            result_string += f"Category {category}: Insufficient data for RMSD calculation.\n"
    #Step 7: calculate CDR RMSD
    cdr_coords_crystal_A = get_atom_coordinates(crystal_pdb, cdr_atoms_cry_A)
    cdr_coords_crystal_B = get_atom_coordinates(crystal_pdb, cdr_atoms_cry_B)
    
    cdr_coords_model_A = []
    cdr_coords_model_B = []

    indices_to_remove_A = []
    indices_to_remove_B = []

    for idx, cdr_coord in enumerate(cdr_coords_crystal_A):
        idx_crystal = np.where(np.all(coordinates_crystal == cdr_coord, axis=1))[0]
        if len(idx_crystal) > 0:
            cdr_coords_model_A.append(y_on_x[idx_crystal[0]])
        else:
            indices_to_remove_A.append(idx)

    for idx, cdr_coord in enumerate(cdr_coords_crystal_B):
        idx_crystal = np.where(np.all(coordinates_crystal == cdr_coord, axis=1))[0]
        if len(idx_crystal) > 0:
            cdr_coords_model_B.append(y_on_x[idx_crystal[0]])
        else:
            indices_to_remove_B.append(idx)

    cdr_coords_crystal_A = np.delete(cdr_coords_crystal_A, indices_to_remove_A, axis=0)
    cdr_coords_crystal_B = np.delete(cdr_coords_crystal_B, indices_to_remove_B, axis=0)

    cdr_coords_model_A = np.array(cdr_coords_model_A)
    cdr_coords_model_B = np.array(cdr_coords_model_B)

    rmsd_cdrs = {}
    if len(cdr_coords_crystal_A) > 0 and len(cdr_coords_model_A) == len(cdr_coords_crystal_A):
        diff = cdr_coords_crystal_A - cdr_coords_model_A
        rmsd = np.sqrt(np.sum(np.square(diff)) / len(cdr_coords_crystal_A))
        rmsd_cdrs['TCRA'] = rmsd
    else:
        rmsd_cdrs['TCRA'] = None
        print("Difference in number of atoms for CDR3 in chain A")
    if len(cdr_coords_crystal_B) > 0 and len(cdr_coords_model_B) == len(cdr_coords_crystal_B):
        diff = cdr_coords_crystal_B - cdr_coords_model_B
        rmsd = np.sqrt(np.sum(np.square(diff)) / len(cdr_coords_crystal_B))
        rmsd_cdrs['TCRB'] = rmsd
    else:
        rmsd_cdrs['TCRB'] = None
        print("Difference in number of atoms for CDR3 in chain B")
    
    result_string+=f"CDR3 TCRA: {rmsd_cdrs['TCRA']:.2f} angstroms\n" if rmsd_cdrs['TCRA'] is not None else "CDR3 TCRA: Insufficient data for RMSD calculation.\n"
    result_string+=f"CDR3 TCRB: {rmsd_cdrs['TCRB']:.2f} angstroms\n" if rmsd_cdrs['TCRB'] is not None else "CDR3 TCRB: Insufficient data for RMSD calculation.\n"
    if len(model_sequences) == 5:
        return (result_string, overall_rmsd, category_rmsd_results["TCRA/TCRB"], category_rmsd_results["Peptide"], category_rmsd_results["MHC/B2M"], rmsd_cdrs.get('TCRA', None), rmsd_cdrs.get('TCRB', None))
    elif len(model_sequences) == 4:
        return (result_string, overall_rmsd, category_rmsd_results["TCRA/TCRB"], category_rmsd_results["Peptide"], category_rmsd_results["MHC"], rmsd_cdrs.get('TCRA', None), rmsd_cdrs.get('TCRB', None))

def run_dockq(model_path, native_path):
    # Ejecutar el comando DockQ
    dockq_command = f"DockQ {model_path} {native_path}"
    result = subprocess.run(dockq_command, shell=True, capture_output=True, text=True, check=True)
    
    # Obtener el texto de salida
    output = result.stdout

    # Usamos expresiones regulares para extraer las métricas
    dockq_score = re.search(r'DockQ:\s*([0-9\.]+)', output)
    irmsd = re.search(r'iRMSD:\s*([0-9\.]+)', output)
    lrmsd = re.search(r'LRMSD:\s*([0-9\.]+)', output)
    fnat = re.search(r'fnat:\s*([0-9\.]+)', output)
    clashes = re.search(r'clashes:\s*(\d+)', output)

    # Parsear los resultados
    dockq_score_val = float(dockq_score.group(1)) if dockq_score else None
    irmsd_val = float(irmsd.group(1)) if irmsd else None
    lrmsd_val = float(lrmsd.group(1)) if lrmsd else None
    fnat_val = float(fnat.group(1)) if fnat else None
    clashes_val = int(clashes.group(1)) if clashes else None

    # Retornar los valores directamente
    return dockq_score_val, irmsd_val, lrmsd_val, fnat_val, clashes_val