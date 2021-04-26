import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")


import itertools
import logging
import unittest
from test.multiprocess_test_case import MultiProcessTestCase, get_random_test_tensor

import crypten
import torch
from crypten.common.tensor_types import is_int_tensor
from crypten.mpc.primitives import BinarySharedTensor
from crypten.mpc import MPCTensor
import crypten.communicator as comm
import time


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import itertools

from torchvision import transforms, datasets, models

from PIL import Image
import numpy as np
import pandas as pd

class MiniONN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        
        self.conv6 = nn.Conv2d(64, 64, 1, 1, 0)
        self.conv7 = nn.Conv2d(64, 16, 1, 1, 0)
        
        self.fc = nn.Linear(1024, 1)
        
        self.avg1 = nn.AvgPool2d(2, 2)
        self.avg2 = nn.AvgPool2d(2, 2)
        
        
    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        
        h = self.avg1(h)
        
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        
        h = self.avg2(h)
        
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        
        h = h.view(-1, 1024)
        h = self.fc(h)
        
        return h
        



class PerformGrater(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        # We don't want the main process (rank -1) to initialize the communcator
        if self.rank >= 0:
            crypten.init()
            
    def test(self):

        model_ft = MiniONN()


        transform = transforms.Compose([
                    lambda x: Image.open(x).convert('RGB'),
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

        if self.rank==0: # ALICE
            
            model = crypten.load('./SCML_Project/training/models/minionn/checkpoint_cpu_cpu.pt', dummy_model=model_ft, src=0)
            dummy_input = torch.empty((1, 3, 32, 32))
            private_model = crypten.nn.from_pytorch(model.double(), dummy_input.double())
            private_model.encrypt(src=0)
            
            
            data_enc = crypten.cryptensor(dummy_input, src=1)
            
            private_model.eval()
            start = time.time()

            output_enc = private_model(data_enc)

            comm.get().send_obj(output_enc.share, 1)
            end = time.time()



        else: # BOB
            model = crypten.load('./SCML_Project/training/models/minionn/minionn_random.pt', dummy_model=model_ft, src=0)
            dummy_input = torch.empty((1, 3, 32, 32))
            private_model = crypten.nn.from_pytorch(model.double(), dummy_input.double())
            private_model.encrypt(src=0)
            
            
            data_enc = crypten.cryptensor(transform('./SCML_Project/training/dataset/COVID/COVID-1.png').unsqueeze(0), src=1)
            
            private_model.eval()
            start = time.time()

            output_enc = private_model(data_enc)

            done = comm.get().recv_obj(0)
            print("Bob received: ", output_enc.share+done)

            end = time.time()
            
        
        print('Time: ', end-start)


if __name__ == "__main__":
    
    unittest.main()