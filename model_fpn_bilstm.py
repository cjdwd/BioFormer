from __future__ import print_function
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import torch.utils.data as Data
import numpy as np

import collections
import math
import copy
torch.manual_seed(1)
np.random.seed(1)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5,
                     stride=stride, padding=2, bias=False)
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv5x5(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.elu(out)
        return out

class fpn_bilstm(torch.nn.Module):
    """
    The class is an implementation of the DrugVQA model including regularization and without pruning. 
    Slight modifications have been done for speedup
    
    """
    def __init__(self,args,block):
        """
        Initializes parameters suggested in paper
 
        args:
            batch_size  : {int} batch_size used for training
            lstm_hid_dim: {int} hidden dimension for lstm
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            n_chars_smi : {int} voc size of smiles
            n_chars_seq : {int} voc size of protein sequence
            dropout     : {float}
            in_channels : {int} channels of CNN block input
            cnn_channels: {int} channels of CNN block
            cnn_layers  : {int} num of layers of each CNN block
            emb_dim     : {int} embeddings dimension
            dense_hid   : {int} hidden dim for the output dense
            task_type   : [0,1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes
 
        Returns:
            self
        """
        super(fpn_bilstm,self).__init__()

        ###drugVQA
        self.batch_size = args['batch_size']
        self.lstm_hid_dim = args['lstm_hid_dim']
        self.r = args['r']
        self.in_channels = args['in_channels']

        #rnn
        self.gene_emb = nn.Embedding(4,args['gene_emb_dim'])
        self.gene_lstm = torch.nn.LSTM(args['gene_emb_dim'], args['gene_hidden_dim'], 2,
                                       batch_first=True, bidirectional=True, dropout=args['dropout'])

        self.gene_bn1 = nn.BatchNorm1d(204800)
        self.gene_linear1 = torch.nn.Linear(args['gene_emb_dim']*204800,args['gene_linear_out_dim'])
        self.gene_bn2 = nn.BatchNorm1d(args['gene_linear_out_dim'])



        self.lstm = torch.nn.LSTM(args['emb_dim'], self.lstm_hid_dim, 2, batch_first=True, bidirectional=True,
                                  dropout=args['dropout'])
        self.embeddings = nn.Embedding(args['n_chars_smi'], args['emb_dim'])
        self.seq_embed = nn.Embedding(args['n_chars_seq'],args['emb_dim'])
        self.lstm = torch.nn.LSTM(args['emb_dim'],self.lstm_hid_dim,2,batch_first=True,bidirectional=True,dropout=args['dropout']) 
        self.linear_first = torch.nn.Linear(2*self.lstm_hid_dim,args['d_a'])
        self.linear_second = torch.nn.Linear(args['d_a'],args['r'])
        self.linear_first_seq = torch.nn.Linear(args['cnn_channels'],args['d_a'])
        self.linear_second_seq = torch.nn.Linear(args['d_a'],self.r)

        #cnn
        self.conv = conv3x3(1, self.in_channels)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.elu = nn.ELU(inplace=False)
        self.layer1 = self.make_layer(block, args['cnn_channels'], args['cnn_layers'])
        self.layer2 = self.make_layer(block, args['cnn_channels'], args['cnn_layers'])

        self.linear_final_step = torch.nn.Linear(self.lstm_hid_dim*2+256,args['dense_hid'])
        self.linear_final = torch.nn.Linear(args['dense_hid'],args['n_classes'])
 
        self.bn1 = nn.BatchNorm1d(545)
        self.bn2 = nn.BatchNorm2d(1)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(args['dense_hid'])
        self.bn5 = nn.BatchNorm1d(args['n_classes'])

        self.ln1 = torch.nn.Linear(384,108)
        self.ln2 = torch.nn.Linear(108,32)
        self.fpn=FPN([3,4,6,3])
        self.sigmoid = nn.Sigmoid()
        self.final_conv = nn.Conv2d(256*4,256,7,bias=False).cuda()

    def softmax(self,input, axis=1):
        """
        Softmax applied to axis=n
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied

        Returns:
            softmaxed tensors
        """
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def get_protein_features(self,x2):
        outputs = self.fpn(x2)
        return outputs

    # x1 is smiles, x2 is contactmap
    def forward(self,x1,x2,hidden_state):

        # extract smiles features
        smile_embed = self.embeddings(x1)         
        outputs, hidden_state = self.lstm(smile_embed,hidden_state)    
        sentence_att = F.tanh(self.linear_first(outputs))       
        sentence_att = self.linear_second(sentence_att)       
        sentence_att = self.softmax(sentence_att,1)       
        sentence_att = sentence_att.transpose(1,2)        
        sentence_embed = sentence_att@outputs
        avg_sentence_embed = torch.sum(sentence_embed,1)/self.r

        # extract contactmap features
        contactmap_outputs = self.fpn(x2)
        contactmap_outputs = self.final_conv(torch.cat((contactmap_outputs[0],contactmap_outputs[1],
                                            contactmap_outputs[2],contactmap_outputs[3]),1)).squeeze()
        contactmap_outputs = self.bn3(contactmap_outputs)
        contactmap_outputs = self.sigmoid(contactmap_outputs)

        sscomplex = torch.cat([avg_sentence_embed,contactmap_outputs],dim=1)
        sscomplex = F.relu(self.bn4(self.linear_final_step(sscomplex)))

        output = F.sigmoid(self.bn5(self.linear_final(sscomplex)))
        return output
 
#bottleneck of resnet
class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,in_planes,planes,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
#         print(downsample)
        self.bottleneck=nn.Sequential(
            nn.Conv2d(in_planes,planes,1,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes,planes,3,stride,1,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes,self.expansion*planes,1,bias=False),
            nn.BatchNorm2d(self.expansion*planes),
        )
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample

    def forward(self,x):
        identity=x
        out=self.bottleneck(x)
        if self.downsample is not None:
#             print(self.downsample)
            identity=self.downsample(x)
        out+=identity
        out=self.relu(out)
        return out

#fpn
class FPN(nn.Module):
    def __init__(self,layers):
        super(FPN,self).__init__()
        self.inplanes=64

        self.conv1=nn.Conv2d(3,64,7,2,3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(3,2,1)

        self.layer1=self._make_layer(64,layers[0])
        self.layer2=self._make_layer(128,layers[1],2)
        self.layer3=self._make_layer(256,layers[2],2)
        self.layer4=self._make_layer(512,layers[3],2)

        self.toplayer=nn.Conv2d(2048,256,1,1,0)

        self.smooth1=nn.Conv2d(256,256,3,1,1)
        self.smooth2=nn.Conv2d(256,256,3,1,1)
        self.smooth3=nn.Conv2d(256,256,3,1,1)

        self.latlayer1=nn.Conv2d(1024,256,1,1,0)
        self.latlayer2=nn.Conv2d(512,256,1,1,0)
        self.latlayer3=nn.Conv2d(256,256,1,1,0)
        
        self.pool2 = nn.MaxPool2d(kernel_size=8,stride=8)
        self.pool3 = nn.MaxPool2d(kernel_size=4,stride=4)
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        
#         self.lateral_conv = Conv2d(512, 256, kernel_size=1, bias=True,
#                       norm=get_norm(norm, out_channels))
        
    def _make_layer(self,planes,blocks,stride=1):
        downsample=None
        if stride!=1 or self.inplanes != Bottleneck.expansion*planes:
            downsample=nn.Sequential(
                nn.Conv2d(self.inplanes,Bottleneck.expansion*planes,1,stride,bias=False),
                nn.BatchNorm2d(Bottleneck.expansion*planes)
            )
        layers=[]
        layers.append(Bottleneck(self.inplanes,planes,stride,downsample))
        self.inplanes=planes*Bottleneck.expansion
        for i in range(1,blocks):
            layers.append(Bottleneck(self.inplanes,planes))
        return nn.Sequential(*layers)


    def _upsample_add(self,x,y):
        _,_,H,W=y.shape
        return F.upsample(x,size=(H,W),mode='bilinear')+y

    def forward(self,x):
        #from bottom to up
        c1=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2=self.layer1(c1)
        c3=self.layer2(c2)
        c4=self.layer3(c3)
        c5=self.layer4(c4)
        #from up to bottom
        p5=self.toplayer(c5)
        p4=self._upsample_add(p5,self.latlayer1(c4))
        p3=self._upsample_add(p4,self.latlayer2(c3))
        p2=self._upsample_add(p3,self.latlayer3(c2))

        p4=self.smooth1(p4)
        p3=self.smooth2(p3)
        p2=self.smooth3(p2)
        return self.pool2(p2),self.pool3(p3),self.pool4(p4),p5,p2,p3,p4