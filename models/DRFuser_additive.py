

import torch
import torch.nn as nn 
import torchvision.models as models 

class Additive_Attention_block(nn.Module):
	def __init__(self,F_g,F_l,F_int):
		super(Additive_Attention_block,self).__init__()
		self.W_g = nn.Sequential(
			nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
			nn.BatchNorm2d(F_int)
			)
		
		self.W_x = nn.Sequential(
			nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
			nn.BatchNorm2d(F_int)
		)

		self.psi = nn.Sequential(
			nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
			nn.BatchNorm2d(1),
			nn.Sigmoid()
		)
		
		self.relu = nn.ReLU(inplace=True)
		
	def forward(self,g,x):
		g1 = self.W_g(g)
		x1 = self.W_x(x)
		psi = self.relu(g1+x1)
		psi = self.psi(psi)

		return x*psi

class DRFuser_additive(nn.Module):

	def __init__(self, args):
		super(DRFuser_additive, self).__init__()

		self.args=args

		self.num_resnet_layers = self.args.resnet

		if self.num_resnet_layers == 18:
			resnet_raw_model1 = models.resnet18(pretrained=True)
			resnet_raw_model2 = models.resnet18(pretrained=True)
			self.inplanes = 512
		elif self.num_resnet_layers == 34:
			resnet_raw_model1 = models.resnet34(pretrained=True)
			resnet_raw_model2 = models.resnet34(pretrained=True)
			self.inplanes = 512
		elif self.num_resnet_layers == 50:
			resnet_raw_model1 = models.resnet50(pretrained=True)
			resnet_raw_model2 = models.resnet50(pretrained=True)
			self.inplanes = 2048
		elif self.num_resnet_layers == 101:
			resnet_raw_model1 = models.resnet101(pretrained=True)
			resnet_raw_model2 = models.resnet101(pretrained=True)
			self.inplanes = 2048
		elif self.num_resnet_layers == 152:
			resnet_raw_model1 = models.resnet152(pretrained=True)
			resnet_raw_model2 = models.resnet152(pretrained=True)
			self.inplanes = 2048

		########  Event ENCODER  ########
 
		# self.encoder_event_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
		self.encoder_event_conv1 = resnet_raw_model1.conv1
		# self.encoder_event_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)
		self.encoder_event_bn1 = resnet_raw_model1.bn1
		self.encoder_event_relu = resnet_raw_model1.relu
		self.encoder_event_maxpool = resnet_raw_model1.maxpool
		self.encoder_event_layer1 = resnet_raw_model1.layer1
		self.encoder_event_layer2 = resnet_raw_model1.layer2
		self.encoder_event_layer3 = resnet_raw_model1.layer3
		self.encoder_event_layer4 = resnet_raw_model1.layer4

		########  RGB ENCODER  ########
 
		self.encoder_rgb_conv1 = resnet_raw_model2.conv1
		self.encoder_rgb_bn1 = resnet_raw_model2.bn1
		self.encoder_rgb_relu = resnet_raw_model2.relu
		self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
		self.encoder_rgb_layer1 = resnet_raw_model2.layer1
		self.encoder_rgb_layer2 = resnet_raw_model2.layer2
		self.encoder_rgb_layer3 = resnet_raw_model2.layer3
		self.encoder_rgb_layer4 = resnet_raw_model2.layer4

		########  Addition  ########
		##Resnet34

		if self.args.resnet == 34:

			self.Att1=Additive_Attention_block(F_g=64,F_l=64,F_int=32)
			self.Att2=Additive_Attention_block(F_g=64,F_l=64,F_int=32)
			self.Att3=Additive_Attention_block(F_g=128,F_l=128,F_int=64)
			self.Att4=Additive_Attention_block(F_g=256,F_l=256,F_int=128)
			self.Att5=Additive_Attention_block(F_g=512,F_l=512,F_int=256)

		elif self.args.resnet ==50:
		##Resnet50

			self.Att1=Additive_Attention_block(F_g=64,F_l=64,F_int=32)
			self.Att2=Additive_Attention_block(F_g=256,F_l=256,F_int=128)
			self.Att3=Additive_Attention_block(F_g=512,F_l=512,F_int=256)
			self.Att4=Additive_Attention_block(F_g=1024,F_l=1024,F_int=512)
			self.Att5=Additive_Attention_block(F_g=2048,F_l=2048,F_int=1024)
		

		########  DECODER  ########
		if self.args.resnet == 34:
			self.join = nn.Sequential(
							nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=0, bias=False), 
							nn.BatchNorm2d(256),
							nn.Dropout(0.2),
							nn.ReLU(),)
		if self.args.resnet == 50:
			self.join = nn.Sequential(
							nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=0, bias=False), 
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0, bias=False), 
							nn.BatchNorm2d(512),
							nn.Dropout(0.2),
							nn.ReLU(),
							nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=0, bias=False), 
							nn.BatchNorm2d(256),
							nn.Dropout(0.2),
							nn.ReLU(),)


 
	def forward(self, event, rgb):

		rgb = rgb
		event = event

		verbose = False

		# encoder

		######################################################################

		if verbose: print("rgb.size() original: ", rgb.size())  # (480, 640)
		if verbose: print("event.size() original: ", event.size()) # (480, 640)

		######################################################################

		rgb = self.encoder_rgb_conv1(rgb)
		if verbose: print("rgb.size() after conv1: ", rgb.size()) # (240, 320)
		rgb = self.encoder_rgb_bn1(rgb)
		if verbose: print("rgb.size() after bn1: ", rgb.size())  # (240, 320)
		rgb = self.encoder_rgb_relu(rgb)
		if verbose: print("rgb.size() after relu: ", rgb.size())  # (240, 320)

		event = self.encoder_event_conv1(event)
		if verbose: print("event.size() after conv1: ", event.size()) # (240, 320)
		event = self.encoder_event_bn1(event)
		if verbose: print("event.size() after bn1: ", event.size()) # (240, 320)
		event = self.encoder_event_relu(event)
		if verbose: print("event.size() after relu: ", event.size())  # (240, 320)

		# rgb = rgb + event
		rgb=self.Att1(g=rgb,x=event)

		rgb = self.encoder_rgb_maxpool(rgb)
		if verbose: print("rgb.size() after maxpool: ", rgb.size()) # (120, 160)

		event = self.encoder_event_maxpool(event)
		if verbose: print("event.size() after maxpool: ", event.size()) # (120, 160)

		######################################################################

		rgb = self.encoder_rgb_layer1(rgb)
		if verbose: print("rgb.size() after layer1: ", rgb.size()) # (120, 160)
		event = self.encoder_event_layer1(event)
		if verbose: print("event.size() after layer1: ", event.size()) # (120, 160)

		# rgb = rgb + event
		rgb=self.Att2(g=rgb,x=event)

		######################################################################
 
		rgb = self.encoder_rgb_layer2(rgb)
		if verbose: print("rgb.size() after layer2: ", rgb.size()) # (60, 80)
		event = self.encoder_event_layer2(event)
		if verbose: print("event.size() after layer2: ", event.size()) # (60, 80)

		# rgb = rgb + event
		rgb=self.Att3(g=rgb,x=event)

		######################################################################

		rgb = self.encoder_rgb_layer3(rgb)
		if verbose: print("rgb.size() after layer3: ", rgb.size()) # (30, 40)
		event = self.encoder_event_layer3(event)
		if verbose: print("event.size() after layer3: ", event.size()) # (30, 40)

		# rgb = rgb + event
		rgb=self.Att4(g=rgb,x=event)

		######################################################################

		rgb = self.encoder_rgb_layer4(rgb)
		if verbose: print("rgb.size() after layer4: ", rgb.size()) # (15, 20)
		event = self.encoder_event_layer4(event)
		if verbose: print("event.size() after layer4: ", event.size()) # (15, 20)

		# fuse = rgb + event   #([2, 2048, 8, 16])
		fuse=self.Att5(g=rgb,x=event)

		######################################################################

		# decoder
		fuse = self.join(fuse)
		# fuse=self.avgpool(fuse)
		fuse=torch.flatten(fuse, 1) 
		self.decoder=nn.Sequential(
							nn.Linear(fuse.shape[1], 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 1),).cuda(self.args.device)

		fuse=self.decoder(fuse)     
		if verbose: print("fuse after join: ", fuse.size()) # (30, 40)


		return fuse



def unit_test():
	import argparse	
	num_minibatch = 2
	parser = argparse.ArgumentParser()
	parser.add_argument('--resnet',  type=int, default=50)
	args = parser.parse_args()
	rgb = torch.randn(num_minibatch, 3, 480, 640)
	event = torch.randn(num_minibatch, 3, 480, 640)
	rtf_net = DRFuser_additive(args)
	input = torch.cat((rgb, event), dim=1)
	rtf_net(event,rgb)


if __name__ == '__main__':
	unit_test()
