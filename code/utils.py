import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolStandardize
from htmd.molecule.util import uniformRandomRotation
from htmd.smallmol.smallmol import SmallMol
from htmd.molecule.voxeldescriptors import _getOccupancyC, _getGridCenters
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
import multiprocessing
import math, os
import random
from htmd.molecule.util import uniformRandomRotation

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

def delNoneMol(smiles,y):

    return smiles, y


def prepDatasets(smiles, np_save_path, y, y_path, length):
    from tqdm import tqdm

    strings = np.zeros((len(smiles), length+2), dtype='uint8')     # 103+2

    vocab_list__ = ["pad", "start", "end",
        "C", "c", "N", "n", "S", "s", "P", "O", "o",
        "B", "F", "I",
        "X", "Y", "Z",
        "1", "2", "3", "4", "5", "6",
        "#", "=", "-", "(", ")",
        "/", "\\", "[", "]", "+", "@", "H", "7"
    ]
    vocab_i2c_v1__ = {i: x for i, x in enumerate(vocab_list__)}
    vocab_c2i_v1__ = {vocab_i2c_v1__[i]: i for i in vocab_i2c_v1__}


    for i, sstring in enumerate(tqdm(smiles)):
        mol = Chem.MolFromSmiles(sstring)
        aa = sstring
        if sstring.find('.') != -1:
            lfc = MolStandardize.fragment.LargestFragmentChooser()
            mol = lfc.choose(mol)

        if not mol:
            raise ValueError("Failed to parse molecule '{}'".format(mol))

        sstring = Chem.MolToSmiles(mol)  # Make the SMILES canonical.
        sstring = sstring.replace("Cl", "X").replace("[nH]", "Y").replace("Br", "Z")
        try:
            vals = [1] + [vocab_c2i_v1__[xchar] for xchar in sstring] + [2]
        except KeyError:
            raise ValueError(("Unkown SMILES tokens: {} in string '{}'."
                              .format(", ".join([x for x in sstring if x not in vocab_c2i_v1__]),
                                                                          sstring)))
        strings[i, :len(vals)] = vals

        if i>999999:
            break

    np.save(np_save_path, strings)
    np.save(y_path, y)


def getRDkitMol(in_smile):
    try:
        m = Chem.MolFromSmiles(in_smile)
        mh = Chem.AddHs(m)
        AllChem.EmbedMolecule(mh)
        Chem.AllChem.MMFFOptimizeMolecule(mh)
        m = Chem.RemoveHs(mh)
        mol = SmallMol(m)
        return mol
    except:
        return None

def readImage(path_list):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img_list = []
    for i in range(len(path_list)):
        img = mpimg.imread(path_list[i])
        img_list.append(img)
    img_arr = np.array(img_list)
    return img_arr

def rotate(coords, rotMat, center=(0,0,0)):
    """ Rotate a selection of atoms by a given rotation around a center """
    newcoords = coords - center
    return np.dot(newcoords, np.transpose(rotMat)) + center

def saveImageFromCoords(coords,i):
    xmax = np.max(coords)
    xmin = np.min(coords)
    ymax = 255
    ymin = 0
    aa = (ymax - ymin) * (coords - xmin) / ((xmax - xmin) + ymin)   #[0,255]
    piexl = np.round(aa)
    image = piexl.astype(int)

    # image = image.astype(int)

    # plt.axis('off')
    # fig = plt.figure(1, figsize=(20, 10), dpi=80)
    # plt.imshow(image)  # Needs to be in row,col order
    # fig.savefig('/media/generate/3D_image/00.jpg')
    #
    #
    # aa1 = (coords - xmin) / ((xmax - xmin) + ymin)                 #[0,1]
    # image1 = np.tile(np.array(aa1), (coords.shape[0], 1))
    # image1 = np.reshape(image1, [coords.shape[0], coords.shape[0], coords.shape[1]])
    # plt.axis('off')
    # fig1 = plt.figure(1, figsize=(20, 10), dpi=80)
    # plt.imshow(image1)  # Needs to be in row,col order
    # fig1.savefig('/media/generate/3D_image/11.jpg')

    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches( 0.608/3, 0.608/3)  # dpi = 300, output = 60*60 pixels 0.608
    # fig.set_size_inches(3.0 / 3, 3.0 / 3)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(image)
    path = '/media/generate/3D_image/'+str(i)+'.jpg'
    fig.savefig(path, format='jpg', transparent=True, dpi=300, pad_inches=0)
    return path

def getImageCoordsAndSaveImage():
    smiles = np.load("../traindataset/smiles.npy")

    if len(smiles)!=0:
        for i in range(5): #len(smiles)
            coords_list = []
            smile = smiles[i]
            smile_str = list(smile)
            end_token = smile_str.index(2)
            smile_str = "".join([vocab_i2c_v1[i] for i in smile_str[1:end_token]])
            mol = getRDkitMol(smile_str)
            if mol is not None:
                coords = mol.getCoords()
                n_atoms = len(coords)
                lig_center = mol.getCenter()

                coords_list.append(coords)
                for p in range(n_atoms-1):
                    rrot = uniformRandomRotation()  # Rotation
                    coords = rotate(coords, rrot, center=lig_center)
                    coords_list.append(coords)

                coords_arr = np.array(coords_list)
                saveImageFromCoords(coords_arr, i)

# getImageCoordsAndSaveImage()

def load_AID1706_SARS_CoV_3CL(path = './data', binary = True, threshold = 15, balanced = True, oversample_num = 30, seed = 1):
    # 2Q6D
    # https://pubchem.ncbi.nlm.nih.gov/bioassay/1706#section=Data-Table
    # https://pubchem.ncbi.nlm.nih.gov/protein/AAZ82016
    # https://www.ncbi.nlm.nih.gov/protein/AAZ82016
    # https://www.uniprot.org/uniprot/Q3Y5H1
    # https://swissmodel.expasy.org/repository/uniprot/Q3Y5H1?csm=63F537111F6025D7
    # https://www.rcsb.org/structure/2q6d

    print('Beginning Processing...')

    if not os.path.exists(path):
        os.makedirs(path)

    target = 'SGFKKLVSPSSAVEKCIVSVSYRGNNLNGLWLGDSIYCPRHVLGKFSGDQWGDVLNLANNHEFEVVTQNGVTLNVVSRRLKGAVLILQTAVANAETPKYKFVKANCGDSFTIACSYGGTVIGLYPVTMRSNGTIRASFLAGACGSVGFNIEKGVVNFFYMHHLELPNALHTGTDLMGEFYGGYVDEEVAQRVPPDNLVTNNIVAWLYAAIISVKESSFSQPKWLESTTVSIEDYNRWASDNGFTPFSTSTAITKLSAITGVDVCKLLRTIMVKSAQWGSDPILGQYNFEDELTPESVFNQVGGVRLQ'

    saved_path_data = "./data/AID_1706_datatable_all.csv"
    saved_path_conversion = "./data/AID1706_training_conversions.csv"

    df_data = pd.read_csv(saved_path_data)
    df_conversion = pd.read_csv(saved_path_conversion)

    val = df_data.iloc[4:][['PUBCHEM_CID','PUBCHEM_ACTIVITY_SCORE']]
    val['binary_label'] = 0
    val['binary_label'][(val.PUBCHEM_ACTIVITY_SCORE >= threshold) & (val.PUBCHEM_ACTIVITY_SCORE <=100)] = 1
    for col in val.columns.values:
        val[col] = val[col].astype('int64')

    cid2smiles = dict(zip(df_conversion[['cid', 'smiles']].values[:, 0], df_conversion[['cid', 'smiles']].values[:, 1]))
    positive_df = val[val.binary_label == 1]
    positive_list = [cid2smiles[i] for i in positive_df.PUBCHEM_CID.values]
    max_lengh_positive_smiles = 0


    for i in range(len(positive_list)):
        if positive_list[i].find("As")!=-1:
            print(positive_list[i])
            print("\n")


    for i in range(len(positive_list)):
        if len(positive_list[i]) > max_lengh_positive_smiles:
            max_lengh_positive_smiles = len(positive_list[i])

    select_val = val[val.PUBCHEM_CID.apply(lambda x: ( len(str(cid2smiles[x]))) <= max_lengh_positive_smiles and str(cid2smiles[x]).find("As")==-1 ) ]   #103

    if balanced:
        a = [select_val[select_val.binary_label==0].sample(n = len(select_val[select_val.binary_label==1]) * oversample_num, replace = False, random_state = seed), pd.concat([select_val[select_val.binary_label==1]]*oversample_num, ignore_index=True)]
        val = pd.concat(a).sample(frac = 1, replace = False, random_state = seed).reset_index(drop = True)

    X_drug = [cid2smiles[i] for i in val.PUBCHEM_CID.values]


    if binary:
        print('Default binary threshold for the binding affinity scores is 15, recommended by the investigator')
        y = val.binary_label.values
    else:
        y = val.PUBCHEM_ACTIVITY_SCORE.values

    return np.array(X_drug), target, np.array(y), max_lengh_positive_smiles

# smiles, X_target, y, max_lengh_positive_smiles = load_AID1706_SARS_CoV_3CL('./data', oversample_num = 30)


def length_func(list_or_tensor):
    if type(list_or_tensor)==list:
        return len(list_or_tensor)
    return list_or_tensor.shape[0]

def protein2emb_encoder(x):
    from DeepPurpose.subword_nmt.subword_nmt.apply_bpe import BPE
    import codecs

    vocab_path = '../../ESPF/protein_codes_uniprot_2000.txt'
    bpe_codes_protein = codecs.open(vocab_path)
    pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
    sub_csv = pd.read_csv('../../ESPF/subword_units_map_uniprot_2000.csv')
    idx2word_p = sub_csv['index'].values
    words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

    max_p = 545
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p

    return i, np.asarray(input_mask)



def protein_process(target_encoding=None):
    target = 'SGFKKLVSPSSAVEKCIVSVSYRGNNLNGLWLGDSIYCPRHVLGKFSGDQWGDVLNLANNHEFEVVTQNGVTLNVVSRRLKGAVLILQTAVANAETPKYKFVKANCGDSFTIACSYGGTVIGLYPVTMRSNGTIRASFLAGACGSVGFNIEKGVVNFFYMHHLELPNALHTGTDLMGEFYGGYVDEEVAQRVPPDNLVTNNIVAWLYAAIISVKESSFSQPKWLESTTVSIEDYNRWASDNGFTPFSTSTAITKLSAITGVDVCKLLRTIMVKSAQWGSDPILGQYNFEDELTPESVFNQVGGVRLQ'

    if target_encoding == 'CNN':
        AA = pd.Series(df_data['Target Sequence'].unique()).apply(trans_protein)
        AA_dict = dict(zip(df_data['Target Sequence'].unique(), AA))
        df_data['target_encoding'] = [AA_dict[i] for i in df_data['Target Sequence']]
    # the embedding is large and not scalable but quick, so we move to encode in dataloader batch.
    elif target_encoding == 'CNN_RNN':
        AA = pd.Series(df_data['Target Sequence'].unique()).apply(trans_protein)
        AA_dict = dict(zip(df_data['Target Sequence'].unique(), AA))
        df_data['target_encoding'] = [AA_dict[i] for i in df_data['Target Sequence']]
    elif target_encoding == 'Transformer':
        coding,mask = protein2emb_encoder(target)
    else:
        raise AttributeError("Please use the correct protein encoding available!")

    return coding,mask


def protein_process_info():
    dataset_id_path = "/home/project/gMol/data/protein/2Q6D_ABC_result"
    L_list = [np.load(dataset_id_path + '/L_' + str(k) + '.npz') for k in range(4)]
    L_list_ = [csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape']) for loader
               in L_list]
    D_list = [np.load(dataset_id_path + '/D_' + str(k) + '.npz') for k in range(3)]
    D_list_ = [csc_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape']) for loader
               in D_list]
    f = np.load(dataset_id_path + '/feature.npy')

    L_ = [torch.Tensor( np.reshape(np.array((L.todense())), (1, L.shape[0], L.shape[1])) ) for L in L_list_]
    D_ = [torch.Tensor( np.reshape(np.array((D.todense())), (1, D.shape[0], D.shape[1])) ) for D in D_list_]
    f_ = torch.Tensor( np.reshape(f, (1, f.shape[0], f.shape[1])) )

    return L_, D_, f_, f.shape[-1]
