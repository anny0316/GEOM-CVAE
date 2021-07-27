# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license

import os
import sys
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import numpy as np

import sys
sys.path.append(r'/home/project/')
from gMol.nets import EncoderCNN, DecoderRNN, GeomCVAE, ChebyNet
from gMol.gene import queue_datagen
from gMol.utils import *
from keras.utils.data_utils import GeneratorEnqueuer
from tqdm import tqdm
import argparse


cap_loss = 0.
caption_start = 5500
batch_size = 16 #128

# savedir = args["output_dir"]
savedir = "model"
os.makedirs(savedir, exist_ok=True)


smiles = np.load("data/smiles_3CL.npy")
y = np.load("data/smiles_3CL_y.npy")

import multiprocessing
multiproc = multiprocessing.Pool(6)
my_gen = queue_datagen(smiles, y, batch_size=batch_size, mp_pool=multiproc)
# my_gen = queue_datagen(smiles, batch_size=batch_size)

mg = GeneratorEnqueuer(my_gen, random_seed=0)
mg.start()
mt_gen = mg.get()
L, D, pfeatures, fshape = protein_process_info()

# Define the networks
encoder = EncoderCNN(3)
decoder = DecoderRNN(512, 1024, 37, 1)
vae_model = GeomCVAE(use_cuda=True)
cheby_model = ChebyNet(fshape, 128, 128, 3, 3)

encoder.cuda()
decoder.cuda()
vae_model.cuda()
cheby_model.cuda()

# Caption optimizer
criterion = nn.CrossEntropyLoss()
caption_params = list(decoder.parameters()) + list(encoder.parameters())
caption_optimizer = torch.optim.Adam(caption_params, lr=0.001)

encoder.train()
decoder.train()

# VAE optimizer
reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False


def loss_function(recon_x, x, mu, logvar, label_y):
    BCE = reconstruction_function(recon_x, x)
    KLD_element = (mu-label_y).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD

vae_cheby_params = list(vae_model.parameters()) + list(cheby_model.parameters())
vae_optimizer = torch.optim.Adam(vae_cheby_params, lr=1e-4)
vae_model.train()
cheby_model.train()

tq_gen = tqdm(enumerate(mt_gen))
# tq_gen = enumerate(mt_gen)
log_file = open(os.path.join(savedir, "log.txt"), "w")
cap_loss = 0.
caption_start = 5500


for i, (mol_batch, condition, y, only_y, caption, lengths) in tq_gen:

    in_data = Variable(mol_batch.cuda())
    # discrim_data = Variable(condition.cuda())
    y_data = Variable(y.cuda())
    p_features = Variable(pfeatures.cuda())
    p_L = [Variable(adj.cuda()) for adj in L]
    p_D = [Variable(d.cuda()) for d in D]
    vae_optimizer.zero_grad()

    c_output1, c_output2 = cheby_model(p_features, p_L, p_D)
    # con = condition.cpu().numpy()

    recon_batch, mu, logvar, label_y = vae_model(in_data, c_output1, c_output2, y_data, only_y)
    vae_loss = loss_function(recon_batch, in_data, mu, logvar, label_y)

    vae_loss.backward(retain_graph=True if i >= caption_start else False)
    p_loss = vae_loss.data
    vae_optimizer.step()

    if i >= caption_start : #caption_start:  # Start by autoencoder optimization
        captions = Variable(caption.cuda())
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        a = caption.numpy()
        b = targets.cpu().numpy()

        decoder.zero_grad()
        encoder.zero_grad()
        features = encoder(recon_batch)
        outputs = decoder(features, captions, lengths)
        cap_loss = criterion(outputs, targets)
        cap_loss.backward()
        caption_optimizer.step()

    if (i + 1) % 6500 == 0:
        torch.save(decoder.state_dict(),
                   os.path.join(savedir,
                                'decoder-%d.pkl' % (i + 1)))
        torch.save(encoder.state_dict(),
                   os.path.join(savedir,
                                'encoder-%d.pkl' % (i + 1)))
        torch.save(vae_model.state_dict(),
                   os.path.join(savedir,
                                'vae-%d.pkl' % (i + 1)))
        torch.save(cheby_model.state_dict(),
                   os.path.join(savedir,
                                'cheby-%d.pkl' % (i + 1)))

    if (i + 1) % 50 == 0:
        result = "Step: {}, caption_loss: {:.5f}, " \
                 "VAE_loss: {:.5f}".format(i + 1,
                                           float(cap_loss.data.cpu().numpy()) if type(cap_loss) != float else 0.,
                                           p_loss)
        log_file.write(result + "\n")
        log_file.flush()
        tq_gen.write(result)

    # Reduce the LR
    if (i + 1) % 9000 == 0:
        # Command = "Reducing learning rate".format(i+1, float(loss.data.cpu().numpy()))
        log_file.write("Reducing LR\n")
        tq_gen.write("Reducing LR")
        for param_group in caption_optimizer.param_groups:
            lr = param_group["lr"] / 2.
            param_group["lr"] = lr

    if i == 13000:
        # We are Done!
        log_file.close()
        break

# Cleanup
del tq_gen
mt_gen.close()
# multiproc.close()
