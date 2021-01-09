#!/home/tony/anaconda3/envs/nn_experiments/bin/python

import numpy as np
import matplotlib.pyplot as plt 


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter

import time
import copy
from collections import defaultdict
import socket    
import os
import datetime


### custom imports

from circle_from_points_dataset import CircleDataset
import helper
from loss import dice_loss

### Reproducability  (Not possible, because of torch.nn.bilinear..)
np.random.seed(1234)
torch.manual_seed(1234)
# torch.set_deterministic(True)  ### biliner interpolation has no deterministic implementation so commenting

hostname = socket.gethostname()


args = dict()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args["device"] = device
print("Training using {}".format(args["device"]))
##### update values ########
args['verbose']=True
args['pid'] = os.getpid()# kill -9 pid
args['cwd'] = os.getcwd().split('/')[-2]### maindir/src (we need maindir)
args['hostname'] = hostname
args['batch_size'] = 16 ## used only for val
args['is_max_pool'] = False
args['n_work'] = 6
args['optimizer_choice'] = "Adam" ### "Adam", "SGD"
args['lr_scheduler'] = "CyclicLR"
args['initialization'] = "kaiming" ## xavier
args['pin'] =  True
args['is_reflect'] = False ## reflect image or not
args['num_class'] = 1
args['tensorboard_logs_dir'] = "/Modeling/pytorch_tensorboard_logs"
args['extra_tag'] = "Expmnt"

run_identifiers = ['cwd','optimizer_choice','lr_scheduler','initialization','extra_tag']

identifier_for_pth = 'Optim{o}_mxplP{m}_lrsc_{r}'.format(o=args['optimizer_choice'],m=args['is_max_pool'],r=args['lr_scheduler'])
run_timestamp = str(datetime.datetime.now()).replace(' ','_')
pth_save_step = 10

run_id = run_timestamp
for k,v in args.items():
	if k in run_identifiers:
		run_id = run_id + '_' + k+'_'+str(v)
run_id = run_id.replace('_','').replace('-','').replace(':','')


args['run_id'] = run_id


print(args)
print("#################################")
## For testing
if hostname == 'tony-TUF-Gaming-FX505GD-FX505GD':
	#tensorboard --logdir=/home/tony/work/pytorch_explore/tensorboard_explore/pytorch_tensorboard_logs
	args['tensorboard_logs_dir'] = "/home/tony/work/pytorch_explore/tensorboard_explore/pytorch_tensorboard_logs"
	args["batch_size"] = 2
	args["n_work"] = 0

##################################
#### load the custom model(according to choice on maxpool)
from model import ResNetUNet

trans = transforms.Compose([
	transforms.ToTensor(),

	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])

target_trans = transforms.Compose([
	transforms.ToTensor(),

])


###################
batch_size=1
num_class = 1
MODEL_PATH = "best_model.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)
print("Infering using {}".format(device))

###################


# #### Model summary
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ResNetUNet(n_class=1)
# model = model.to(device)
def reverse_transform(inp):
	inp = inp.numpy().transpose((1, 2, 0))
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	inp = std * inp + mean
	inp = np.clip(inp, 0, 1)
	inp = (inp * 255).astype(np.uint8)
	
	return inp

def evalModel(model):
	for input, labels in dataloaders['val']:
		input = input.to(device, dtype=torch.float)

		labels = labels.to(device, dtype=torch.float)
		start_time = time.time()
		outputs = model(input)
		torch.cuda.synchronize()
		print(time.time()-start_time)

		
		pred = outputs.cpu().detach().numpy()[0,:,:,:].transpose(1,2,0)
		print(pred.shape)
		pred[pred>=0.5]=1
		pred[pred<0.5]=0



		print((labels.shape,outputs.shape))
		f, axarr = plt.subplots(3)
		f.set_size_inches(18.5, 10.5)
		axarr[0].imshow(input.cpu().detach().numpy()[0,:,:,:].transpose(1,2,0))
		axarr[0].axis('off')
		axarr[1].imshow(labels.cpu().detach().numpy()[0,:,:,:].transpose(1,2,0),cmap='gray')
		axarr[1].axis('off')
		# axarr[2].imshow(input.cpu().detach().numpy()[0,:,:,:].transpose(1,2,0)*labels.cpu().detach().numpy()[0,:,:,:].transpose(1,2,0),cmap='gray')


		axarr[2].imshow(pred,cmap='gray')
		axarr[2].axis('off')




		plt.show()










####Define the main training loop
from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss


if __name__ == "__main__":

	train_set = CircleDataset(image_count=200,image_size=320,pt_thickness=10,transform=trans,target_transform=target_trans)
	val_set = CircleDataset(image_count=100,image_size=320,pt_thickness=10,transform=trans,target_transform=target_trans)
	print("train val dataset size {},{}".format(len(train_set),len(val_set)))


	# global image_datasets

	image_datasets = {
		'train': train_set, 'val': val_set
	}




	dataloaders = {
		'train': DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['n_work'], pin_memory=args['pin']),
		'val': DataLoader(val_set, batch_size=args['batch_size'], shuffle=False, num_workers=args['n_work'], pin_memory=args['pin'])
	}



	
	model = ResNetUNet(args['num_class']).to(device)

	model.load_state_dict(torch.load(MODEL_PATH))

	model = model.to(device)
	inp = torch.randn((2,3,320,320)).to(device)
	model(inp)

	evalModel(model)
