import os
import subprocess
from Bio.PDB import MMCIFParser, PDBIO, is_aa, PDBParser, PPBuilder
import uuid

residue_mapping = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

def cif_to_pdb(cif_file):
    """
    Converts a CIF file to PDB format using Biopython's MMCIFParser and PDBIO.
    :param cif_file: Path to the input CIF file
    :saves the converted PDB file in the same directory with the same base name and .pdb extension
    """
    # Define the output PDB file path (same folder, same name, .pdb extension)
    pdb_file = os.path.splitext(cif_file)[0] + ".pdb"
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(os.path.basename(pdb_file), cif_file)
    except Exception as e:
        print(f"Error parsing CIF file {cif_file}: {e}")
        return None
    io = PDBIO()
    io.set_structure(structure)
    try:
        io.save(pdb_file)
        print(f"Successfully converted {cif_file} to {pdb_file}")
    except Exception as e:
        print(f"Error writing PDB file {pdb_file}: {e}")

def merge_pdb(pdb_file, chain_mapping = {"tcra_chain": "D", "tcrb_chain": "E", "mhc_chain": "A", "b2_chain": "B", "peptide_chain": "C"}):
    """
    Merges specified chains from a PDB file into a single PDB file with two chains:
    - Chain A: MHC, B2M, and peptide
    - Chain B: TCR alpha and beta chains
    :param pdb_file: Path to the input PDB file
    :param chain_mapping: Dictionary mapping component names to their respective chain IDs in the input PDB file
    :saves the merged PDB file in the same directory with the same base name and "_merged.pdb" extension
    """
    # Define chain IDs
    tcra_id = chain_mapping.get("tcra_chain", "D")
    tcrb_id = chain_mapping.get("tcrb_chain", "E")
    mhc_id = chain_mapping.get("mhc_chain", "A")
    b2_id = chain_mapping.get("b2_chain", "B")
    epitope_id = chain_mapping.get("peptide_chain", "C")
    
    # Extract base name and define output file path
    base_name = os.path.splitext(os.path.basename(pdb_file))[0]
    dir_name = os.path.dirname(pdb_file)
    output_file_path = os.path.join(dir_name, f"{base_name}_merged.pdb")

    # Preprocess the input file to remove invalid lines and save a temporary cleaned file
    cleaned_pdb_file = os.path.join(dir_name, f"{base_name}_cleaned.pdb")
    cleaned_lines = remove_headers(pdb_file)
    with open(cleaned_pdb_file, 'w') as cleaned_file:
        cleaned_file.writelines(cleaned_lines)
    
    run_id = uuid.uuid4().hex[:8]

    A_tmp = os.path.join(dir_name, f"A_{run_id}.pdb")
    B_tmp = os.path.join(dir_name, f"B_{run_id}.pdb")

    command_AB = (f"pdb_selchain -{tcra_id},{tcrb_id} {cleaned_pdb_file} "
                f"| pdb_chain -B | pdb_reres -1 | pdb_delhetatm > {B_tmp}")

    command_MB = (f"pdb_selchain -{mhc_id},{b2_id},{epitope_id} {cleaned_pdb_file} "
                f"| pdb_chain -A | pdb_reres -1 | pdb_delhetatm > {A_tmp}")

    try:
        subprocess.run(command_MB, shell=True, check=True)
        subprocess.run(command_AB, shell=True, check=True)

        A_lines = remove_headers(A_tmp)
        B_lines = remove_headers(B_tmp)

        with open(output_file_path, "w") as outfile:
            outfile.writelines(A_lines)
            outfile.writelines(B_lines)

        os.remove(A_tmp)
        os.remove(B_tmp)
        os.remove(cleaned_pdb_file)

        print("Successfully merged chains", output_file_path)

    except subprocess.CalledProcessError as e:
        print(f"Error processing {pdb_file}: {e}")

def remove_headers(file_path):
    """
    Reads a PDB file and returns a list of lines that are ATOM records with sufficient length.
    :param file_path: Path to the PDB file
    :return: List of cleaned lines containing only ATOM records
    """
    cleaned_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            # Keep only ATOM lines with sufficient length
            if (line.startswith("ATOM")) and len(line) > 21:
                cleaned_lines.append(line)
    return cleaned_lines

def extract_sequences(file_path):
    """
    Extracts sequences from a PDB or CIF file and returns them as strings and tuples.
    """
    if file_path.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    elif file_path.endswith(".cif") or file_path.endswith(".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Parsear la estructura
    structure = parser.get_structure("structure", file_path)

    sequences_str = {}
    sequences_tuples = {}

    for model in structure:
        for chain in model:
            chain_id = chain.id
            seq_str = []
            seq_tuples = []
            for residue in chain:
                if is_aa(residue, standard=True):
                    res_name = residue.get_resname()
                    resid = residue.get_id()[1]
                    seq_str.append(residue_mapping.get(res_name, 'X'))
                    seq_tuples.append((res_name, resid))
            sequences_str[chain_id] = ''.join(seq_str)
            sequences_tuples[chain_id] = seq_tuples

    return sequences_str, sequences_tuples

def extract_residues_and_resids(pdb_file, chain_id):
    """
    Extracts residue information from a PDB file for a specified chain.
    :param pdb_file: Path to the input PDB file
    :param chain_id: ID of the chain to extract residues from
    :return: A list of tuples containing residue IDs and one-letter codes
    """
    if pdb_file.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    elif pdb_file.endswith(".cif") or pdb_file.endswith(".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file format: {pdb_file}")
    structure = parser.get_structure('structure', pdb_file)    
    residues = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    resid = residue.get_id()[1]
                    resname = residue.get_resname()
                    resname = residue.get_resname().upper()
                    residue_one_letter =  residue_mapping.get(resname, 'X')  # Use 'X' for unknown residues
                    residues.append((resid, residue_one_letter))  
    return residues



def get_seq_from_pdb_chain(pdb_file, chain_id):
    """
    Extracts the amino acid sequence from a specified chain in a PDB file.
    :param pdb_file: Path to the input PDB file
    :param chain_id: ID of the chain to extract the sequence from
    :return: The amino acid sequence as a string for the specified chain, or an empty string if the chain is not found in the PDB file
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]
    
    ppb = PPBuilder()
    actual_seq = ""
    if chain_id in model:
        for pp in ppb.build_peptides(model[chain_id]):
            actual_seq += str(pp.get_sequence())
    return actual_seq