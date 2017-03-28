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
import pandas as pd
from pybrain import SigmoidLayer,LSTMLayer,SoftmaxLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

SETTINGS = {'trainingDataFile': 'numerai_training_data.csv',
		'tournamentDataFile': 'numerai_tournament_data.csv',
		'hiddenclass': LSTMLayer,
		'recurrent': True,
		'outclass': SigmoidLayer,
		'bias': True,
		'IL': 50,
		'HL': 5,
		'OL': 1,
		'continueEpochs': 2,
		'validationProportion': 0.05,
		'maxEpochs': None,
		'verbose': True,
		'fileoutput':''}

def makename():
	name = "%d-%d-%d-B%s-cE%s-mE%s-%R%s_preditions.csv"   % (str(SETTINGS['IL']),str(SETTINGS['HL']),
						str(SETTINGS['OL']),str(SETTINGS['bias']),str(SETTINGS['continueEpochs']),
						str(SETTINGS['maxEpochs']),
						str(SETTINGS['recurrent']))
	return name

NET = buildNetwork(SETTINGS['IL'], SETTINGS['HL'], SETTINGS['OL'], bias=SETTINGS['bias'], hiddenclass=SETTINGS['hiddenclass'],recurrent=SETTINGS['recurrent'],outclass=SETTINGS['outclass'])
DATASET = SupervisedDataSet(SETTINGS['IL'], SETTINGS['OL'])

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
	results = []
	csv = load_csv(filename)
	for sample in csv:
		id = sample[0]		
		inputs = sample[1:]
		results.append([id,NET.activate(inputs)[0]])
	return results

def save_result(filename,data):
	print "saving:",filename
	fp = open(filename,'w')
	fp.write('t_id,probability\n')
	for row in data:
		fp.write(','.join([str(x) for x in row])+'\n')

def train_network():
    """
    Trains the network.
    """
    print 'Training network, please wait ...'
    trainer = BackpropTrainer(NET,verbose=SETTINGS['verbose'])
    trainer.trainUntilConvergence(dataset=DATASET, maxEpochs=SETTINGS['maxEpochs'], verbose=SETTINGS['verbose'],
                                  continueEpochs=SETTINGS['continueEpochs'], validationProportion=SETTINGS['validationProportion'])

	
if __name__ == "__main__":
	load_training_data(SETTINGS['trainingDataFile'])
	train_network()       
	results = load_tournament_data(SETTINGS['tournamentDataFile'])
	save_result(makename(),results)
