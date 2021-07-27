# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


########################
# AutoEncoder Networks #
########################

class GeomCVAE(nn.Module):
    def __init__(self, nc=3, ndf=128, latent_variable_size=512, use_cuda=False):
        super(GeomCVAE, self).__init__()
        self.use_cuda = use_cuda
        self.nc = nc
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)


        self.e2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)

        self.e3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.e4 = nn.Conv2d(64, 512, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)

        self.e5 = nn.Conv2d(512, 512, 3, 2, 0)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(512 * 3 * 3, latent_variable_size)
        self.fc2 = nn.Linear(512 * 3 * 3, latent_variable_size)
        self.label = nn.Linear(2, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, 512 * 3 * 3)

        # up5
        self.d2 = nn.ConvTranspose2d(512, 512, 3, 2, padding=0, output_padding=1)
        self.bn6 = nn.BatchNorm2d(512, 1.e-3)

        # up 4
        self.d3 = nn.ConvTranspose2d(512, 256, 3, 2, padding=1, output_padding=0)
        self.bn7 = nn.BatchNorm2d(256, 1.e-3)

        # up3 12 -> 12
        self.d4 = nn.ConvTranspose2d(256, 128, 3, 2, padding=1, output_padding=1)
        self.bn8 = nn.BatchNorm2d(128, 1.e-3)

        # up2 12 -> 24
        self.d5 = nn.ConvTranspose2d(128, 32, 3, 2, padding=1, output_padding=1)
        self.bn9 = nn.BatchNorm2d(32, 1.e-3)

        # Output layer
        self.d6 = nn.Conv2d(32, nc, 3, 1, padding=1)

        # Condtional encoding
        # self.ce1 = nn.Conv2d(3, 32, 3, 1, 1)
        # self.ce2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.cond_c2 = nn.Linear(268, 15*15)
        self.cond_c1 = nn.Linear(1330, 30*30)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, y):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, 512 * 3 * 3)
        return self.fc1(h5), self.fc2(h5), self.label(y)

    def reparametrize(self, mu, logvar, factor):
        std = logvar.mul(0.5).exp_()
        if self.use_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return (eps.mul(std) * factor).add_(mu)

    def decode(self, z, c1, c2, only_y):
        # Conditional block
        # cc1 = self.relu(self.ce1(cond_x))
        # cc2 = self.relu(self.ce2(cc1))
        c1 = c1.view(c1.shape[1], c1.shape[2])
        c2 = c2.view(c2.shape[1], c2.shape[2])
        c1_condition_f, c2_condition_f = [], []
        c1_negative_f = np.zeros((c1.shape[0],c1.shape[1]), dtype=np.float32)
        c2_negative_f = np.zeros((c2.shape[0],c2.shape[1]), dtype=np.float32)
        c1_negative_f = torch.Tensor(c1_negative_f)
        c2_negative_f = torch.Tensor(c2_negative_f)
        c1_negative_f = Variable(c1_negative_f.cuda())
        c2_negative_f = Variable(c2_negative_f.cuda())
        for i in range(len(only_y)):
            if only_y[i] == 0:
                c1_condition_f.append(c1_negative_f)
                c2_condition_f.append(c2_negative_f)
            elif only_y[i] == 1:
                c1_condition_f.append(c1)
                c2_condition_f.append(c2)

        c1_condition = torch.stack(c1_condition_f, 0)
        c2_condition = torch.stack(c2_condition_f, 0)


        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, 512, 3, 3)
        h2 = self.leakyrelu(self.bn6(self.d2((h1))))

        h3 = self.leakyrelu(self.bn7(self.d3(h2)))
        cc2 = c2_condition.view(c2_condition.shape[0], h3.shape[1], -1)
        h3_ = h3.view(h3.shape[0],h3.shape[1],-1)
        cat_h3 = torch.cat([h3_, cc2], dim=2)
        cat_h3 = cat_h3.view(-1,cat_h3.shape[2])
        cat_h3 = self.relu(self.cond_c2(cat_h3))
        cat_h3 = cat_h3.view(-1,15,15)
        h3 = cat_h3.view(h3.shape[0],h3.shape[1],15,15)

        h4 = self.leakyrelu(self.bn8(self.d4(h3)))
        cc1 = c1_condition.view(c1_condition.shape[0], h4.shape[1], -1)
        h4_ = h4.view(h4.shape[0],h4.shape[1],-1)
        cat_h4 = torch.cat([h4_, cc1], dim=2)
        cat_h4 = cat_h4.view(-1, cat_h4.shape[2])
        cat_h4 = self.relu(self.cond_c1(cat_h4))
        cat_h4 = cat_h4.view(-1, 30, 30)
        h4 = cat_h4.view(h4.shape[0],h4.shape[1],30,30)

        # h4 = torch.cat([h4, cc2], dim=1)
        h5 = self.leakyrelu(self.bn9(self.d5(h4)))
        # h5 = torch.cat([h5, cc1], dim=1)
        return self.sigmoid(self.d6(h5))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view())
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x, c1, c2, y, only_y, factor=1.):
        mu, logvar, label_y = self.encode(x, y)
        z = self.reparametrize(mu, logvar, factor=factor)
        yx = y.cpu().numpy()
        res = self.decode(z, c1, c2, only_y)
        return res, mu, logvar, label_y


class EncoderCNN(nn.Module):
    def __init__(self, in_layers):
        super(EncoderCNN, self).__init__()
        self.pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()
        layers = []
        out_layers = 32

        layers.append(nn.Conv2d(in_layers, 32, 3, stride=1, bias=False, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(self.relu)
        layers.append(self.pool)
        layers.append(nn.Conv2d(32, 64, 3, stride=1, bias=False, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(self.relu)
        layers.append(self.pool)
        layers.append(nn.Conv2d(64, 128, 3, stride=1, bias=False, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(self.relu)
        layers.append(self.pool)
        layers.append(nn.Conv2d(128, 256, 3, stride=1, bias=False, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(self.relu)
        layers.append(self.pool)

        # layers.pop()  # Remove the last max pooling layer!
        self.fc1 = nn.Linear(256, 512)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        x = x.mean(dim=2).mean(dim=2)
        x = self.relu(self.fc1(x))
        return x


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers): #(512,1024,29,1)
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)#(29,512)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True) #(512,1024,1)
        self.linear = nn.Linear(hidden_size, vocab_size)#(1024,29)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """Decode shapes feature vectors and generates SMILES."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Samples SMILES tockens for given shape features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(105):  #62
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            a = outputs.max(1)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids

    def sample_prob(self, features, states=None):
        """Samples SMILES tockens for given shape features (probalistic picking)."""
        sampled1_ids, sampled2_ids = [], []
        inputs = features.unsqueeze(1)
        for i in range(105):  # 62  maximum sampling length
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            if i == 0:
                predicted = outputs.max(1)[1]
            else:
                probs = F.softmax(outputs, dim=1)

                # Probabilistic sample tokens
                if probs.is_cuda:
                    probs_np = probs.data.cpu().numpy()
                else:
                    probs_np = probs.data.numpy()

                rand_num = np.random.rand(probs_np.shape[0])
                iter_sum = np.zeros((probs_np.shape[0],))
                tokens = np.zeros(probs_np.shape[0], dtype=np.int)

                for i in range(probs_np.shape[1]):
                    c_element = probs_np[:, i]
                    iter_sum += c_element
                    valid_token = rand_num < iter_sum
                    update_indecies = np.logical_and(valid_token,
                                                     np.logical_not(tokens.astype(np.bool)))
                    tokens[update_indecies] = i

                # put back on the GPU.
                if probs.is_cuda:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)).cuda())
                else:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)))

            sampled1_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        inputs = features.unsqueeze(1)
        for i in range(62):  # 62  maximum sampling length
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            if i == 0:
                predicted = outputs.max(1)[1]
            else:
                probs = F.softmax(outputs, dim=1)

                # Probabilistic sample tokens
                if probs.is_cuda:
                    probs_np = probs.data.cpu().numpy()
                else:
                    probs_np = probs.data.numpy()

                rand_num = np.random.rand(probs_np.shape[0])
                iter_sum = np.zeros((probs_np.shape[0],))
                tokens = np.zeros(probs_np.shape[0], dtype=np.int)

                for i in range(probs_np.shape[1]):
                    c_element = probs_np[:, i]
                    iter_sum += c_element
                    valid_token = rand_num < iter_sum
                    update_indecies = np.logical_and(valid_token,
                                                     np.logical_not(tokens.astype(np.bool)))
                    tokens[update_indecies] = i

                # put back on the GPU.
                if probs.is_cuda:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)).cuda())
                else:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)))

            sampled2_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        return sampled1_ids, sampled2_ids


class ChebyNet(nn.Module):
    """ Chebyshev Network, see reference below for more information
        Defferrard, M., Bresson, X. and Vandergheynst, P., 2016.
        Convolutional neural networks on graphs with fast localized spectral filtering. In NIPS.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer, chebyshev_order):
        super(ChebyNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layer = num_layer
        self.polynomial_order = chebyshev_order
        # self.dropout = config.model.dropout if hasattr(config.model, 'dropout') else 0.0
        self.dropout = 0.0

        dim_list = [self.input_dim, self.hidden_dim, self.hidden_dim, self.output_dim]
        self.filter = nn.ModuleList([ nn.Linear(dim_list[tt]*(self.polynomial_order+1), dim_list[tt + 1]) for tt in range(self.num_layer)] + [nn.Linear(dim_list[-2], dim_list[-1])])
        # self.embedding = nn.Embedding(self.num_atom, self.input_dim)

        # attention
        self.att_func = nn.Sequential(*[nn.Linear(dim_list[-1], 1), nn.Sigmoid()])

        self._init_param()

    def _init_param(self):
        for ff in self.filter:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()

        for ff in self.att_func:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()

    def forward(self, node_feat, L_, D_, label=None, mask=None):
        """
          shape parameters:
            batch size = B
            embedding dim = D
            max number of nodes within one mini batch = N
            number of edge types = E
            number of predicted properties = P

          Args:
            node_feat: long tensor, shape B X N
            L: float tensor, shape B X N X N X (E + 1)
            label: float tensor, shape B X P
            mask: float tensor, shape B X N
        """
        batch_size = node_feat.shape[0]
        num_node = node_feat.shape[1]
        state = node_feat  # shape: B X N X D
        outputhidden = node_feat

        # propagation
        # Note: we assume adj = 2 * L / lambda_max - I
        for tt in range(self.num_layer):
            L=L_[tt]
            state_scale = [None] * (self.polynomial_order + 1)
            state_scale[-1] = state
            state_scale[0] = torch.bmm(L, state)
            for kk in range(1, self.polynomial_order):
                state_scale[kk] = 2.0 * torch.bmm(L, state_scale[kk - 1]) - state_scale[kk - 2]

            msg = torch.cat(state_scale, dim=2).view(num_node * batch_size, -1)
            state = F.relu(self.filter[tt](msg)).view(batch_size, num_node, -1)
            pooling_state = torch.bmm(D_[tt], state)
            state = F.dropout(pooling_state, self.dropout, training=self.training)
            num_node = state.shape[1]

            if tt == 1:
                outputhidden = state

        # output
        state = state.view(batch_size * num_node, -1)
        y = self.filter[-1](state)  # shape: BN X 1
        att_weight = self.att_func(state)  # shape: BN X 1
        y = (att_weight * y).view(batch_size, num_node, -1)

        return outputhidden, y
