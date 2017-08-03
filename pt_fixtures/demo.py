import csv
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.hidden = nn.Linear(2, 2)
		self.hidden2 = nn.Linear(2, 2)
		self.drop = nn.Dropout(p=0.5)
		self.hidden3 = nn.Linear(2, 2)
		self.hidden4 = nn.Linear(2, 2)
		self.drop1 = nn.Dropout(p=0.5)
		self.out = nn.Linear(2, 1)

	def forward(self, x):
		x = F.relu(self.hidden(x))
		x = F.relu(self.hidden2(x))
		x = F.relu(self.drop(x))
		x = F.relu(self.hidden3(x))
		x = F.relu(self.hidden4(x))
		x = F.relu(self.drop1(x))
		x = self.out(x)
		return x

model = Net()

for i in range(1):
	with open('PT_94.csv', 'r') as fixtures:
		spamreader = csv.reader(fixtures, delimiter=' ', quotechar='|')
		for row in spamreader:
			content = row[0].split(',')
			print('home team: ' + content[0] + ' away team: ' + content[1])
			print('result: ' + content[2] + '-' + content[3])

			x = torch.Tensor(2, 2)
			x[0][0] = int(content[0])
			x[1][0] = int(content[1])
			x[0][1], x[1][1] = 1, 0
			input = Variable(x, requires_grad=False)
			result = model(input)
			true_result = torch.Tensor(2,1)
			true_result[0] = int(content[2])
			true_result[1] = int(content[3])
			target = Variable(true_result, requires_grad=False)
			optimizer = optim.SGD(model.parameters(), lr=0.001)
			criterion = nn.L1Loss()
			optimizer.zero_grad()
			loss = criterion(result, target)
			loss.backward()
			optimizer.step()
			print(loss)

with open('PT_95.csv', 'r') as prediction:
	spamreader = csv.reader(prediction, delimiter=' ', quotechar='|')
	totalright = 0
	size = 0
	for row in spamreader:
		game = row[0].split(',')
		x = torch.Tensor(2, 2)
		x[0][0] = int(game[0])
		x[1][0] = int(game[1])
		x[0][1], x[1][1] = 1, 0
		input = Variable(x, requires_grad=True)
		predict = model(input).data.numpy()
		size += 1
		if((predict[0] > predict[1]) and (game[2] > game[3])):
			print('got it right')
			totalright += 1
		if((predict[0] < predict[1]) and (game[2] < game[3])):
			print('got it right') 
			totalright +=1

		print('result was: ' + game[2] + '-' + game[3])
		print('predicted: ' + str(predict[0]) + '-' + str(predict[1]))

	print('totalright: ', totalright)
	print('size of sample: ', size)
	print('hit rate: ', totalright/size)