

import torch
import torch.nn as nn 
import torchvision.models as models 
import torch.nn.functional as F

from models.attention import AttentionConv, AttentionStem


class DRFuser_self_attention(nn.Module):

	def __init__(self, args ):
		super(DRFuser_self_attention, self).__init__()

		self.args=args
		self.num_resnet_layers = args.resnet

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

		########  Self-attention  ########

		if self.args.resnet == 34:

			self.Att1=AttentionConv(in_channels=64,out_channels=64,kernel_size=3,padding=1)
			self.Att2=AttentionConv(in_channels=64,out_channels=64,kernel_size=3,padding=1)
			self.Att3=AttentionConv(in_channels=128,out_channels=128,kernel_size=3,padding=1)
			self.Att4=AttentionConv(in_channels=256,out_channels=256,kernel_size=3,padding=1)
			self.Att5=AttentionConv(in_channels=512,out_channels=512,kernel_size=3,padding=1)
		elif self.args.resnet == 50:
			self.Att1=AttentionConv(in_channels=64,out_channels=64,kernel_size=3,padding=1)
			self.Att2=AttentionConv(in_channels=256,out_channels=256,kernel_size=3,padding=1)
			self.Att3=AttentionConv(in_channels=512,out_channels=512,kernel_size=3,padding=1)
			self.Att4=AttentionConv(in_channels=1024,out_channels=1024,kernel_size=3,padding=1)
			self.Att5=AttentionConv(in_channels=2048,out_channels=2048,kernel_size=3,padding=1)

		########  DECODER  ########
		
		if self.args.resnet == 34:
			self.join = nn.Sequential(
							nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=0, bias=False), 
							nn.BatchNorm2d(256),
							nn.Dropout(0.2),
							nn.ReLU(),)
		elif self.args.resnet ==50:					
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

		rgb = self.Att1(rgb) + self.Att1(event)
	

		rgb = self.encoder_rgb_maxpool(rgb)
		if verbose: print("rgb.size() after maxpool: ", rgb.size()) # (120, 160)

		event = self.encoder_event_maxpool(event)
		if verbose: print("event.size() after maxpool: ", event.size()) # (120, 160)

		######################################################################

		rgb = self.encoder_rgb_layer1(rgb)
		if verbose: print("rgb.size() after layer1: ", rgb.size()) # (120, 160)
		event = self.encoder_event_layer1(event)
		if verbose: print("event.size() after layer1: ", event.size()) # (120, 160)

		rgb = self.Att2(rgb) + self.Att2(event)


		######################################################################
 
		rgb = self.encoder_rgb_layer2(rgb)
		if verbose: print("rgb.size() after layer2: ", rgb.size()) # (60, 80)
		event = self.encoder_event_layer2(event)
		if verbose: print("event.size() after layer2: ", event.size()) # (60, 80)

		rgb = self.Att3(rgb) + self.Att3(event)
	

		######################################################################

		rgb = self.encoder_rgb_layer3(rgb)
		if verbose: print("rgb.size() after layer3: ", rgb.size()) # (30, 40)
		event = self.encoder_event_layer3(event)
		if verbose: print("event.size() after layer3: ", event.size()) # (30, 40)

		rgb = self.Att4(rgb) + self.Att4(event)
	

		######################################################################

		rgb = self.encoder_rgb_layer4(rgb)
		if verbose: print("rgb.size() after layer4: ", rgb.size()) # (15, 20)
		event = self.encoder_event_layer4(event)
		if verbose: print("event.size() after layer4: ", event.size()) # (15, 20)

		fuse = self.Att5(rgb) + self.Att5(event)   #([2, 2048, 8, 16])
	

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
	rtf_net = DRFuser_self_attention(args)
	
	rtf_net(event,rgb)
	#print('The model: ', rtf_net.modules)

if __name__ == '__main__':
	unit_test()
