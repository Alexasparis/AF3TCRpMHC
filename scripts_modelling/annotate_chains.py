#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
import argparse
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

MEMORY = "5G"

def run_mir(pdb_dir,
            mir_path="./src/mir-1.0-SNAPSHOT.jar",
            output_dir="mir_output",
            arg="annotate-structures",  
            print_log=True):
    
    pdb_dir = Path(pdb_dir)
    if not pdb_dir.exists():
        raise ValueError(f"El directorio {pdb_dir} no existe.")

    pdb_list = list(pdb_dir.glob("*"))
    if not pdb_list:
        raise ValueError(f"No se encontraron archivos en {pdb_dir}")

    pdb_paths = " ".join(str(p) for p in pdb_list)

    # Crear el directorio de salida
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    cmd = f"java -Xmx{MEMORY} -cp {mir_path} com.milaboratory.mir.scripts.Examples {arg} -I {pdb_paths} -O {output_dir_path}/"

    try:
        result = subprocess.run(cmd, shell=True, check=True,
                                stdout=(None if print_log else subprocess.DEVNULL),
                                stderr=(None if print_log else subprocess.DEVNULL))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute '{cmd}'") from e

def main():
    parser = argparse.ArgumentParser(description="Run MIR annotation on PDB files")
    parser.add_argument("--pdb_dir", type=str, default="../data/20250820250813_PDB_TCRpMHC_classI/",
                        help="File with PDBs (default: ../20250813_pdb_tcrpmhc/)")
    parser.add_argument("--output_dir", type=str, default="./data/structures_annotation/",
                        help="Output dir (default: ./data/structures_annotation/)")
    parser.add_argument("--mir_path", type=str, default="../src/mir-1.0-SNAPSHOT.jar",
                        help="MIR script (default: ./src/mir-1.0-SNAPSHOT.jar)")
    parser.add_argument("--arg", type=str, default="annotate-structures",
                        help="Mir action (default: annotate-structures)")
    parser.add_argument("--no_log", action="store_true", help="No mostrar logs de ejecución")
    
    args = parser.parse_args()

    # Verificar versión de Java
    subprocess.run("java -version", shell=True)

    run_mir(pdb_dir=args.pdb_dir,
            output_dir=args.output_dir,
            mir_path=args.mir_path,
            arg=args.arg,
            print_log=not args.no_log)
    
    general_file = Path(args.output_dir) / "general.txt"
    if general_file.exists():
        df = pd.read_csv(general_file, sep="\t")
        df["pdb.id"] = df["pdb.id"].str.replace(".pdb", "", regex=False)
        df.to_csv(general_file, sep="\t", index=False)
    else:
        print(f"Not found {general_file}")

if __name__ == "__main__":
    main()