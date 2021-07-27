from gMol.comgen import CompoundGenerator
import os
import torch

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole

# t1 = Chem.MolFromSmiles("C1=CC=C(C=C1)CCN(CCC2=CC=CC=C2)C(=O)CCl")
# Chem.Draw.MolToFile(t1,'111.png')


my_gen = CompoundGenerator(use_cuda=True)
vae_weights =  os.path.join("model/", "vae-6500.pkl")
encoder_weights =  os.path.join("model/", "encoder-6500.pkl")
decoder_weights = os.path.join("model/", "decoder-6500.pkl")
cheby_weights = os.path.join("model/", "cheby-6500.pkl")

my_gen.load_weight(vae_weights, encoder_weights, decoder_weights, cheby_weights)

seed_mol = "CC1=CC=C(C=C1)C(=O)OC2=CC3=C(C=C2)N(C=N3)C"     # active  y=1

gen_mols = my_gen.generate_molecules(seed_mol,
                                     n_attemps=20,  # How many attemps of generations will be carried out
                                     lam_fact=1.,  # Variability factor
                                     probab=True,  # Probabilistic RNN decoding
                                     filter_unique_valid=False)  # Filter out invalids and replicates

seed = Chem.MolFromSmiles(seed_mol)
# Chem.Draw.MolToFile(seed, 'genValidMol/active_seed.png')
# Chem.Draw.MolsToGridImage(gen_mols)
