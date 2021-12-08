import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
from torch.autograd import Variable
import torchvision.utils as utils
from models.DRFuser import DRFuser
from data.event_dataloader import EventDataset as EV
from torchvision import transforms, utils, models
from sklearn.metrics import mean_squared_error
torch.cuda.empty_cache()
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='self-Attention', help='self-Attention, No-Attention, Additive.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--root_dir', type=str, default='/media/shoaib/work/dataset/dvs_data/', help='Path')
parser.add_argument('--csv_file', type=str, default='//media/shoaib/work/dataset/dvs_data/train_data_our.csv', help='Path')
parser.add_argument('--data_name', type=str, default='drfuser', help='ddd,eventscape,drfuser')
parser.add_argument('--epochs', type=int, default=100, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
parser.add_argument('--val_every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('--num_workers', '-j', type=int, default=0)
parser.add_argument('--image_channel', type=int, default=3)
parser.add_argument('--image_size',  type=int, default=256)
parser.add_argument('--resnet',  type=int, default=50)

args = parser.parse_args()
args.logdir = os.path.join(args.logdir, args.model_id)

writer = SummaryWriter(log_dir=args.logdir)


class Engine(object):
	"""Engine that runs training and inference.
	Args
		- cur_epoch (int): Current epoch.
		- print_every (int): How frequently (# batches) to print loss.
		- validate_every (int): How frequently (# epochs) to run validation.
		
	"""

	def __init__(self,  cur_epoch=0, cur_iter=0):
		self.cur_epoch = cur_epoch
		self.cur_iter = cur_iter
		self.bestval_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.bestval = 1e10
		self.msc = []
		self.true_angle=[]
		self.pred_angle=[]
	

	def train(self):
		loss_epoch = 0.
		num_batches = 0
		model.train()

		# Train loop
		for training_sample in tqdm(train_loader):
				
			# efficiently zero gradients
			for p in model.parameters():
					p.grad = None
			
			# create batch and move to GPU
			dvs_image = training_sample['dvs_image'].to(args.device, dtype=torch.float32)
			aps_image = training_sample['aps_image'].to(args.device, dtype=torch.float32)
			angle = training_sample['angle'].to(args.device, dtype=torch.float32)
			
			
			pred_a = model(dvs_image,aps_image)
			
			
			
			loss = F.l1_loss(pred_a.squeeze(1), angle, reduction='none').mean()

			loss.backward()
			loss_epoch += float(loss.item())

			num_batches += 1
			optimizer.step()
			self.cur_iter += 1
		
		
		loss_epoch = loss_epoch / num_batches
		self.train_loss.append(loss_epoch)
		self.cur_epoch += 1
		writer.add_scalar('train_loss', loss_epoch, self.cur_epoch)
		print('Train Loss:',loss_epoch)
		


	def validate(self):
		model.eval()

		with torch.no_grad():	
			num_batches = 0
			wp_epoch = 0.

			# Validation loop
			for batch_num, test_samples in enumerate(tqdm(test_loader), 0):
				
				# create batch and move to GPU
				dvs_image = test_samples['dvs_image'].to(args.device, dtype=torch.float32)
				aps_image = test_samples['aps_image'].to(args.device, dtype=torch.float32)
				angle = test_samples['angle'].to(args.device, dtype=torch.float32)
				

				# target point

				pred_a = model( dvs_image,aps_image)
			
				wp_epoch += float(F.l1_loss(pred_a.squeeze(1), angle, reduction='none').mean())

				num_batches += 1
					
			wp_loss = wp_epoch / float(num_batches)
			tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' loss: {wp_loss:3.3f}')

			writer.add_scalar('val_loss', wp_loss, self.cur_epoch)
			
			self.val_loss.append(wp_loss)

	def save(self):

		save_best = False
		if self.val_loss[-1] <= self.bestval:
			self.bestval = self.val_loss[-1]
			self.bestval_epoch = self.cur_epoch
			save_best = True
		
		# Create a dictionary of all data to save
		log_table = {
			'epoch': self.cur_epoch,
			'iter': self.cur_iter,
			'bestval': self.bestval,
			'bestval_epoch': self.bestval_epoch,
			'train_loss': self.train_loss,
			'val_loss': self.val_loss,
		}

		# Save ckpt for every epoch
		torch.save(model.state_dict(), os.path.join(args.logdir, 'model_%d.pth'%self.cur_epoch))

		# Save the recent model/optimizer states
		torch.save(model.state_dict(), os.path.join(args.logdir, 'model.pth'))
		torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))

		# Log other data corresponding to the recent model
		# with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
		# 	f.write(json.dumps(log_table))

		tqdm.write('====== Saved recent model ======>')
		
		if save_best:
			torch.save(model.state_dict(), os.path.join(args.logdir, 'best_model.pth'))
			torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
			tqdm.write('====== Overwrote best model ======>')

	def test(self):
		model.eval()

		with torch.no_grad():	
			
			epoch_error = 0.

			# Validation loop
			for batch_num, test_samples in enumerate(tqdm(test_loader), 0):
				
				# create batch and move to GPU
				dvs_image = test_samples['dvs_image'].to(args.device, dtype=torch.float32)
				aps_image = test_samples['aps_image'].to(args.device, dtype=torch.float32)
				angle = test_samples['angle'].to(args.device, dtype=torch.float32)
				

				# target point

				pred_a = model( dvs_image,aps_image)
				error=mean_squared_error(angle.cpu().numpy(), pred_a.cpu().numpy()[0])
				self.msc.append(error)
				epoch_error=(epoch_error+error)/(batch_num+1)
				self.true_angle.append(angle.cpu().numpy()[0])
				self.pred_angle.append(pred_a.cpu().numpy()[0])

	def plot(self):	
		plt.plot(self.msc,'r')
		plt.xlabel('num of images')
		plt.ylabel('error')
		plt.savefig('error.png')
		plt.figure().clear()	
		plt.plot(self.true_angle,'b')
		plt.plot(self.pred_angle,'r')
		plt.xlabel('num of images')
		plt.ylabel('angle')
		plt.savefig('angle.png')
			
			



# Data
event_dataset = EV(args, csv_file=args.csv_file,root_dir=args.root_dir,transform=transforms.Compose([transforms.ToTensor()]),)
dataset_size = int(len(event_dataset))
del event_dataset
split_point = int(dataset_size * 0.8)

train_dataset = EV(args,csv_file=args.csv_file,root_dir=args.root_dir,transform=transforms.Compose([transforms.ToTensor()]),select_range=(0,split_point))
test_dataset = EV(args,csv_file=args.csv_file,root_dir=args.root_dir,transform=transforms.Compose([transforms.ToTensor()]),select_range=(split_point,dataset_size))


train_loader  = DataLoader(
	dataset     = train_dataset,
	batch_size  = args.batch_size,
	shuffle     = True,
	num_workers = args.num_workers,
	pin_memory  = True,
	drop_last   = True
)

test_loader = DataLoader(
	dataset      = test_dataset,
	batch_size   = args.batch_size,
	shuffle      = False,
	num_workers  = args.num_workers,
	pin_memory   = True,
	drop_last    = True
)
# Model


model = DRFuser.build(args)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
trainer = Engine()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print ('Total trainable parameters: ', params)

# Create logdir
if not os.path.isdir(args.logdir):
	os.makedirs(args.logdir)
	print('Created dir:', args.logdir)
elif os.path.isfile(os.path.join(args.logdir, 'recent.log')):
	print('Loading checkpoint from ' + args.logdir)

	# Load checkpoint
	model.load_state_dict(torch.load(os.path.join(args.logdir, 'best_model.pth')))
	optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'best_optim.pth')))

# Log args
with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
	json.dump(args.__dict__, f, indent=2)

for epoch in range(trainer.cur_epoch, args.epochs): 
	trainer.train()
	if epoch % args.val_every == 0: 
		trainer.validate()
		trainer.save()
trainer.test()
trainer.plot()