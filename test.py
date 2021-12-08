import argparse
import json
import os
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, dataloader
from torchvision import transforms, utils, models
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from models.DRFuser import DRFuser

from data.test_event_loader import EventDataset as EV

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='self-Attention', help='self-Attention, No-Attention, Additive.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--root_dir', type=str, default='/home/farzeen/dvs_data/', help='Path')
parser.add_argument('--csv_file', type=str, default='/home/farzeen/dvs_data/train_data_our.csv', help='Path')
parser.add_argument('--data_name', type=str, default='drfuser', help='ddd,eventscape,drfuser')
parser.add_argument('--epochs', type=int, default=100, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
parser.add_argument('--val_every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('--num_workers', '-j', type=int, default=0)
parser.add_argument('--image_channel', type=int, default=3)
parser.add_argument('--image_size',  type=int, default=256)
parser.add_argument('--resnet',  type=int, default=50)

args = parser.parse_args()
args.logdir = os.path.join(args.logdir, args.model_id)



class Engine(object):
	"""Engine that runs training and inference.
	Args
		- cur_epoch (int): Current epoch.
		- print_every (int): How frequently (# batches) to print loss.
		- validate_every (int): How frequently (# epochs) to run validation.
		
	"""

	def __init__(self,  cur_epoch=0, cur_iter=0):
		self.prediction = []
		self.prediction_plot = []
		self.actual = []
		self.actual_plot = []
		self.error = []
		self.dvs_filename = []
		self.aps_filename = []
		self.bestval = 1e10

	def test(self):
		model.eval()

		with torch.no_grad():	

			# Validation loop
			for batch_num, test_samples in enumerate(tqdm(dataloader_test), 0):
				
				# create batch and move to GPU
				dvs_image = test_samples['dvs_image'].to(args.device, dtype=torch.float32)
				rgb_image = test_samples['aps_image'].to(args.device, dtype=torch.float32)
				angle = test_samples['angle'].to(args.device, dtype=torch.float32)
				# speed = test_samples['speed'].to(args.device, dtype=torch.float32)
				dvs_file = str(test_samples['dvs_filename'])[2:-2]
				aps_file = str(test_samples['aps_filename'])[2:-2]
				self.dvs_filename.append(dvs_file)
				self.aps_filename.append(aps_file)
				# target point

				
				pred_angle = model(dvs_image,rgb_image)
				# pred_angle = torch.unsqueeze(pred_angle,0)
				self.prediction.append(str(pred_angle.cpu().numpy()[0][0]))
				self.prediction_plot.append(pred_angle.cpu().numpy()[0])
				self.actual.append(str(angle.cpu().numpy()[0]))
				self.actual_plot.append(angle.cpu().numpy()[0])
				err = mean_squared_error(pred_angle.cpu(),angle.cpu())
				self.error.append(err)
			
				
	def save(self):
		result_dict = {'APS_filename':self.aps_filename,'DVS_filename':self.dvs_filename,'True_angle':self.actual,'Predicted_angle':self.prediction}
		pandas_dataframe = pd.DataFrame(result_dict)
		pandas_dataframe.to_csv('ResNet34-RGB-our1.csv')

	def plot(self):	
		# np.savetxt('pred.csv',self.prediction)
		# np.savetxt('true_angle.csv',self.actual)
		err = mean_squared_error(self.actual_plot,self.prediction_plot)
		print('RMSE:',math.sqrt(err))
		plt.plot(self.actual_plot,'b',label="true")
		plt.plot(self.prediction_plot,'r',label="predicted")	
		plt.xlabel('num of images')
		plt.ylabel('angle')
		plt.legend(loc="upper left")
		plt.savefig('best-self-attention-ResNet-34.png')
	



test_set = EV(args,csv_file=args.csv_file,root_dir=args.root_dir,transform=transforms.Compose([transforms.ToTensor()]),)
dataloader_test = DataLoader(
    test_set, batch_size=args.batch_size, shuffle=False, drop_last=True,num_workers=args.num_workers, pin_memory=True)



model = DRFuser.build(args)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
trainer = Engine()
	# Load checkpoint
model.load_state_dict(torch.load(os.path.join(args.logdir, 'model.pth')))
print('Loading checkpoint from ' + args.logdir)
	# optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'best_optim.pth')))

trainer.test()
trainer.plot()
trainer.save()
print('done')

