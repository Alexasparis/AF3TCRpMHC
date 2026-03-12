import subprocess
import re
from Bio.PDB import MMCIFParser, PDBParser
from utils import residue_mapping

def run_anarci(sequence):
    """
    Runs the ANARCI tool on a given amino acid sequence to obtain IMGT numbering.
    :param sequence: Amino acid sequence as a string
    :return: Output from ANARCI as a string, or an error message if the command fails
    """
    try:
        command=f"ANARCI -i {sequence} --scheme imgt"
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def parse_anarci_output(anarci_output):
    """
    Parses the output from ANARCI to extract IMGT numbering and corresponding residues.
    :param anarci_output: Output from ANARCI as a string
    :return: A list of tuples containing IMGT numbers and corresponding residues, ensuring uniqueness of IMGT numbers
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
    Maps IMGT numbering to original PDB residue IDs by aligning the sequences and accounting for gaps in the IMGT numbering.
    :param imgt_numbered_seq: List of tuples containing IMGT numbers and corresponding residues
    :param pdb_resids: List of tuples containing original PDB residue IDs and corresponding residues
    :return: A list of tuples containing original PDB residue IDs, IMGT numbers, and corresponding residues, where original PDB residue IDs are None for gaps in the IMGT numbering
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
    """
    Parses the CDR3 region from the mapped IMGT numbering and original PDB residue IDs.
    :param mapping: List of tuples containing original PDB residue IDs, IMGT numbers, and corresponding residues
    :return: A list of tuples containing original PDB residue IDs, IMGT numbers, and corresponding residues for the CDR3 region, where original PDB residue IDs are None for gaps in the IMGT numbering
    """
    cdr3_tuples = [tupple for tupple in mapping if 104 <= tupple[1] <= 118 and tupple[2] != "-"]
    return cdr3_tuples

def parse_CDR2 (mapping):
    """Parses the CDR2 region from the mapped IMGT numbering and original PDB residue IDs.
    :param mapping: List of tuples containing original PDB residue IDs, IMGT numbers, and corresponding residues
    :return: A list of tuples containing original PDB residue IDs, IMGT numbers, and corresponding residues for the CDR2 region, where original PDB residue IDs are None for gaps in the IMGT numbering
    """
    cdr2_tuples = [tupple for tupple in mapping if 56 <= tupple[1] <= 65 and tupple[2] != "-"]
    return cdr2_tuples

def parse_CDR1 (mapping):
    """Parses the CDR1 region from the mapped IMGT numbering and original PDB residue IDs.
    :param mapping: List of tuples containing original PDB residue IDs, IMGT numbers, and corresponding residues
    :return: A list of tuples containing original PDB residue IDs, IMGT numbers, and corresponding residues for the CDR1 region, where original PDB residue IDs are None for gaps in the IMGT numbering
    """
    cdr1_tuples = [tupple for tupple in mapping if 27 <= tupple[1] <= 38 and tupple[2] != "-"]
    return cdr1_tuples

def extract_atoms_for_cdr(cdr_list, pdb_file, chain_id):
    """
    Extracts atom information for the specified CDR residues from the given PDB file and chain.
    :param cdr_list: List of tuples containing original PDB residue IDs, IMGT numbers, and corresponding residues for the CDR region
    :param pdb_file: Path to the input PDB file
    :param chain_id: ID of the chain to extract atoms from
    :return: A list of tuples containing atom names, original PDB residue IDs, residue names, and chain IDs for the atoms in the specified CDR region
    """
    if pdb_file.endswith(".pdb"):
        parser =PDBParser(QUIET=True)
    else:
        parser=MMCIFParser(QUIET=True)
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