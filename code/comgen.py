# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license

from gMol.nets import EncoderCNN, DecoderRNN, GeomCVAE, ChebyNet
from gMol.gene import *
from gMol.gene import queue_datagen
from gMol.utils import *
from keras.utils.data_utils import GeneratorEnqueuer
from tqdm import tqdm
from rdkit import Chem
from torch.autograd import Variable
import torch
import time

vocab_list = ["pad", "start", "end",
              "C", "c", "N", "n", "S", "s", "P", "O", "o",
              "B", "F", "I",
              "Cl", "[nH]", "Br", # "X", "Y", "Z",
              "1", "2", "3", "4", "5", "6",
              "#", "=", "-", "(", ")",  # Misc
              "/","\\", "[", "]", "+", "@", "H", "7"
]
vocab_i2c_v1 = {i: x for i, x in enumerate(vocab_list)}
vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}

def decode_smiles(in_tensor1, in_tensor2):
    """
    Decodes input tensor to a list of strings.
    :param in_tensor:
    :return:
    """
    gen_smiles = []
    for sample in in_tensor1:
        csmile = ""
        for xchar in sample[1:]:
            if xchar == 2:
                break
            csmile += vocab_i2c_v1[xchar]
        gen_smiles.append(csmile)

    for sample in in_tensor2:
        csmile = ""
        for xchar in sample[1:]:
            if xchar == 2:
                break
            csmile += vocab_i2c_v1[xchar]
        gen_smiles.append(csmile)

    return gen_smiles

def filter_unique_canonical(in_mols):
    """
    :param in_mols - list of SMILES strings
    :return: list of uinique and valid SMILES strings in canonical form.
    """
    xresults = [Chem.MolFromSmiles(x) for x in in_mols]  # Convert to RDKit Molecule
    xresults = [Chem.MolToSmiles(x) for x in xresults if x is not None]  # Filter out invalids
    return [Chem.MolFromSmiles(x) for x in set(xresults)]  # Check for duplicates and filter out invalids


def get_mol_voxels(smile_str):
    """
    Generate voxelized representation of a molecule.
    :param smile_str: string - molecule represented as a string
    :return: list of torch.Tensors
    """
    # Convert smile to 3D structure
    mol = generate_representation(smile_str)
    if mol is None:
        return None

    # Generate sigmas
    sigmas, coords, lig_center = generate_sigmas(mol)
    vox = torch.Tensor(voxelize(sigmas, coords, lig_center))
    return vox[:5], vox[5:]

def generate_image(smile):
    mol = getRDkitMol(smile)

    coords_list = []
    if mol is not None:
        coords = mol.getCoords()
        n_atoms = len(coords)
        lig_center = mol.getCenter()

        coords_list.append(coords)
        for p in range(n_atoms - 1):
            rrot = uniformRandomRotation()  # Rotation
            coords = rotate(coords, rrot, center=lig_center)
            coords_list.append(coords)

        coords_arr = np.array(coords_list)

        aa = time.clock()
        path = saveImageFromCoords(coords_arr, i=time.clock())
        img = mpimg.imread(path)
        img = img/255
        img = img.astype(np.float32).transpose(2, 0, 1)

    return torch.Tensor(img), torch.Tensor(img)

def generate_batch_image(smile, y):
    smile_str = list(smile)
    end_token = smile_str.index(2)
    smile_str = "".join([vocab_i2c_v1[i] for i in smile_str[1:end_token]])
    mol = getRDkitMol(smile_str)

    coords_list = []
    if mol is not None:
        coords = mol.getCoords()
        n_atoms = len(coords)
        lig_center = mol.getCenter()

        coords_list.append(coords)
        for p in range(n_atoms - 1):
            rrot = uniformRandomRotation()  # Rotation
            coords = rotate(coords, rrot, center=lig_center)
            coords_list.append(coords)

        coords_arr = np.array(coords_list)
        path = saveImageFromCoords(coords_arr, i=time.clock())
        img_ = mpimg.imread(path)
        img_ = img_/255
        img_ = img_.astype(np.float32).transpose(2, 0, 1)
        return img_, end_token + 1
    else:
        return None

class CompoundGenerator:
    def __init__(self, use_cuda=True):

        self.use_cuda = False
        self.encoder = EncoderCNN(3)
        self.decoder = DecoderRNN(512, 1024, 37, 1)
        self.vae_model = GeomCVAE(use_cuda=use_cuda)
        self.cheby_model = ChebyNet(29, 128, 128, 3, 3)

        self.vae_model.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.cheby_model.eval()

        if use_cuda:
            assert torch.cuda.is_available()
            self.encoder.cuda()
            self.decoder.cuda()
            self.vae_model.cuda()
            self.cheby_model.cuda()
            self.use_cuda = True

    def load_weight(self, vae_weights, encoder_weights, decoder_weights, cheby_weights):
        """
        Load the weights of the models.
        :param vae_weights: str - VAE model weights path
        :param encoder_weights: str - captioning model encoder weights path
        :param decoder_weights: str - captioning model decoder model weights path
        :return: None
        """
        self.cheby_model.load_state_dict(torch.load(cheby_weights, map_location='cpu'))
        self.vae_model.load_state_dict(torch.load(vae_weights, map_location='cpu'))
        self.encoder.load_state_dict(torch.load(encoder_weights, map_location='cpu'))
        self.decoder.load_state_dict(torch.load(decoder_weights, map_location='cpu'))


    def caption_shape(self, in_shapes, probab=False):
        """
        Generates SMILES representation from in_shapes
        """
        embedding = self.encoder(in_shapes)
        if probab:
            captions1,captions2 = self.decoder.sample_prob(embedding)
        else:
            captions = self.decoder.sample(embedding)

        captions1 = torch.stack(captions1, 1)
        captions2 = torch.stack(captions2, 1)
        if self.use_cuda:
            captions1 = captions1.cpu().data.numpy()
            captions2 = captions2.cpu().data.numpy()
        else:
            captions1 = captions1.data.numpy()
            captions2 = captions2.data.numpy()
        return decode_smiles(captions1,captions2)

    def generate_molecules(self, smile_str, n_attemps=300, lam_fact=1., probab=False, filter_unique_valid=True):
        """
        Generate novel compounds from a seed compound.
        :param smile_str: string - SMILES representation of a molecule
        :param n_attemps: int - number of decoding attempts
        :param lam_fact: float - latent space pertrubation factor
        :param probab: boolean - use probabilistic decoding
        :param filter_unique_canonical: boolean - filter for valid and unique molecules
        :return: list of RDKit molecules.
        """

        shape_input, cond_input = generate_image(smile_str)
        if self.use_cuda:
            shape_input = shape_input.cuda()
            cond_input = cond_input.cuda()

        shape_input = shape_input.unsqueeze(0).repeat(n_attemps, 1, 1, 1)
        cond_input = cond_input.unsqueeze(0).repeat(n_attemps, 1, 1, 1)

        shape_input = Variable(shape_input, volatile=True)
        cond_input = Variable(cond_input, volatile=True)

        L, D, pfeatures, fshape = protein_process_info()
        p_features = Variable(pfeatures.cuda())
        p_L = [Variable(adj.cuda()) for adj in L]
        p_D = [Variable(d.cuda()) for d in D]
        c_output1, c_output2 = self.cheby_model(p_features, p_L, p_D)


        y_data = np.tile([0,1],n_attemps)
        only_y = np.ones((n_attemps,),dtype=int)
        only_y = tuple(only_y)
        y_data = y_data.reshape((n_attemps,2))
        y_data = Variable(torch.Tensor(y_data).cuda())

        recoded_shapes, _, _, _ = self.vae_model(shape_input, c_output1, c_output2, y_data, only_y, lam_fact)
        smiles = self.caption_shape(recoded_shapes, probab=probab)
        if filter_unique_valid:
            return filter_unique_canonical(smiles)

        t = 0
        for j in range(len(smiles)):
            mol = Chem.MolFromSmiles(smiles[j])
            if mol != None:
                t = t + 1
                Chem.Draw.MolToFile(mol, 'genValidMol/seed2/9/active_' + str(t) + '.png')
                print(smiles[j])
        print(t)

        return [Chem.MolFromSmiles(x) for x in smiles]


    def my_generate_molecules(self, smiles_list, n_attemps=300, lam_fact=1., probab=False, filter_unique_valid=True):
        """
        Generate novel compounds from a seed compound.
        :param smile_str: string - SMILES representation of a molecule
        :param n_attemps: int - number of decoding attempts
        :param lam_fact: float - latent space pertrubation factor
        :param probab: boolean - use probabilistic decoding
        :param filter_unique_canonical: boolean - filter for valid and unique molecules
        :return: list of RDKit molecules.
        """
        vmol, fmol, canonical_smiles_list = [],[],[]
        for i in range(len(smiles_list)):
            shape_input, cond_input = generate_image(smiles_list[i])
            if self.use_cuda:
                shape_input = shape_input.cuda()
                cond_input = cond_input.cuda()

            shape_input = shape_input.unsqueeze(0).repeat(n_attemps, 1, 1, 1)
            cond_input = cond_input.unsqueeze(0).repeat(n_attemps, 1, 1, 1)

            shape_input = Variable(shape_input, volatile=True)
            cond_input = Variable(cond_input, volatile=True)

            L, D, pfeatures, fshape = protein_process_info()
            p_features = Variable(pfeatures.cuda())
            p_L = [Variable(adj.cuda()) for adj in L]
            p_D = [Variable(d.cuda()) for d in D]
            c_output1, c_output2 = self.cheby_model(p_features, p_L, p_D)


            y_data = np.tile([0,1],n_attemps)
            only_y = np.ones((n_attemps,),dtype=int)
            only_y = tuple(only_y)
            y_data = y_data.reshape((n_attemps,2))
            y_data = Variable(torch.Tensor(y_data).cuda())

            recoded_shapes, _, _, _ = self.vae_model(shape_input, c_output1, c_output2, y_data, only_y, lam_fact)
            smiles = self.caption_shape(recoded_shapes, probab=probab)
            t = 0
            for j in range(len(smiles)):
                mol = Chem.MolFromSmiles(smiles[j])
                if mol != None:
                    t = t + 1
                    vmol.append(mol)
                    Chem.Draw.MolToFile(mol, 'ss/'+str(i)+'/active_' + str(t) + '.png')
                    print(smiles[j])
            print(t)


            if filter_unique_valid:
                mol_smiles = filter_unique_canonical(smiles)
                for m in range(len(mol_smiles)):
                    fmol.append(mol_smiles[m])

            seed = Chem.MolFromSmiles(smiles_list[i])
            Chem.Draw.MolToFile(seed, 'ss/'+str(i)+'/active_seed.png')

        print("result:")
        print(len(vmol)/len(fmol))
        print("len")
        print(len(vmol))
        print(len(fmol))


        for x in set(fmol):
            canonical_smiles_list.append(Chem.MolToSmiles(x,isomericSmiles=True))

        print(len(canonical_smiles_list))

        s = []
        smiles = np.load("../data/smiles_3CL.npy")
        for z in range(len(smiles)):
            smile_str = list(smiles[z])
            end_token = smile_str.index(2)
            smile_str = "".join([vocab_i2c_v1[i] for i in smile_str[1:end_token]])
            s.append(smile_str)
        new_smiles = [Chem.MolFromSmiles(x) for x in s]
        new_smiles = [Chem.MolToSmiles(x) for x in new_smiles if x is not None]

        count = 0
        for p in range(len(canonical_smiles_list)):
            for q in range(len(new_smiles)):
                if(canonical_smiles_list[p].strip() == new_smiles[q].strip()):
                    count = count+1

        print(count)
        return vmol, fmol
