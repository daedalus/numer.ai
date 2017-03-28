#!/usr/bin/python
"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
from PIL import Image
from pybrain import SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer


NET = buildNetwork(50, 20, 1, bias=True, hiddenclass=SigmoidLayer) 
DATASET = SupervisedDataSet(50, 1)


def load_csv(filename):
	csv = []
	fp = open(filename)
	c = 0
	for line in fp:
		if c > 0:
			csv.append(line.rstrip().split(','))
		c+=1
	print "regs:",c
	return csv
		

def load_training_data(filename):
	print "loading:",filename
	csv = load_csv(filename)
	for sample in csv:
		inputs = sample[:-1:]		
		output = sample[-1:]
		DATASET.addSample(inputs,output)

def load_tournament_data(filename):
	print "loading:",filename
	results = ['it_id','probability']
	csv = load_csv(filename)
	for sample in csv:
		id = sample[0]		
		inputs = sample[1:]
		results.append([id,NET.activate(inputs)])
	return results

def save_result(filename,data):
	print "saving:",filename
	fp = open(filename,'w')
	for line in data:
		fp.writeline(data.join(',')+'\n')
	

def train_network():
    """
    Trains the network.
    """
    print 'Training network, please wait ...'
    trainer = BackpropTrainer(NET,verbose=True)
    trainer.trainUntilConvergence(dataset=DATASET, maxEpochs=1, verbose=True,
                                  continueEpochs=1, validationProportion=0.025)
 

load_training_data('numerai_training_data.csv')
train_network()       
results = load_tournament_data('numerai_tournament_data.csv')
save_result('result.csv',results)
