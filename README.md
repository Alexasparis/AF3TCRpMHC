# AF3TCRpMHC
This repositoy contains scripts and examples to process AF3 generated TCR-pMHC class I complexes and compare them to their corresponding crystal structures.

## Requirements

- Python 3.6 or later.

Required Python packages:

- pandas==2.2.3
- biopython==1.81
- numpy==1.26.4
- anarci==1.3
- pdb-tools==2.5.0

To install them run: 

```bash
pip install -r requirements.txt
```

### Additional requirements (to convert mmCIF files into pdb):
Either clone and install BeEM from: https://github.com/kad-ecoli/BeEM.git
```bash
git clone https://github.com/kad-ecoli/BeEM.git && cd BeEM && pip install .
```
```bash
conda install -c conda-forge -c schrodinger pymol-bundle
pip install pymol-open-source
```

## Structure of the repo
The project follows the following directory structure:

```bash
strucTCR/
│
├── structures/                   # Structures files
├── README.md                     # Project documentation
└──requirements.txt               # Python dependencies
```
