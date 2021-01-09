#!/home/tony/anaconda3/envs/nn_experiments/bin/python


# #https://github.com/usuyama/pytorch-unet/blob/master/pytorch_resnet18_unet.ipynb

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
args['initialization'] = None #"kaiming" ## xavier
args['pin'] =  True
args['is_reflect'] = False ## reflect image or not
args['num_class'] = 1
args['tensorboard_logs_dir'] = "tensorboard_path"
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
writer = SummaryWriter(args['tensorboard_logs_dir'])
#### load the custom model(according to choice on maxpool)
from model import ResNetUNet
# from model import ResNetUNetNoSkip  ### skip connection has high chance of assuming identity function



### transformations ###
trans = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])

target_trans = transforms.Compose([
	transforms.ToTensor(),

])

####################################

def calc_loss(pred, target, metrics, bce_weight=0.5):

	"""
	Validate F.binary_cross_entropy_with_logits(pred, target, reduction='none').mean() and 
	F.binary_cross_entropy_with_logits(pred, target, reduction='mean') are same:
	Rsult : Validated binary_cross_entropy_with_logits_test.py
	"""
	bce = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')


	pred = torch.sigmoid(pred)
	dice = dice_loss(pred, target)

	loss = bce * bce_weight + dice * (1 - bce_weight)

	metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
	metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
	metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

	return loss


def train_model(model, optimizer, scheduler, num_epochs,dataloaders):
	best_epoch_loss = np.inf
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		since = time.time()

		# Each epoch has a training and validation phase

		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			metrics = defaultdict(float)
			epoch_samples = 0

			k = 0 # there should only be one param_group
			for param_group in optimizer.param_groups:
				epoch_lr = param_group['lr']
				if k>0:
					exit("need to check multiple param group")
				k += 1
				print("LR", param_group['lr'])

			for input, labels in dataloaders[phase]: ## labels is mask
				# # print(phase)

				# ######### visualize #########
				# print((input.shape,labels.shape))
				# # print((input1[0,:].permute(1,2,0).shape,input2[0,:].permute(1,2,0).shape,labels[0,:].permute(1,2,0)[:,:,0].shape))
				# f, axarr = plt.subplots(2)
				# axarr[0].imshow(input[0,:].permute(1,2,0))
				# # axarr[1,0].imshow(input2[0,:].permute(1,2,0))
				# axarr[1].imshow(labels[0,:].permute(1,2,0)[:,:,0],cmap='gray')
				# # axarr[3].imshow(labels[0,:].permute(1,2,0)[:,:,2],cmap='gray')
				# plt.show()
				# continue

				# ##########################

				
				# use the created array to output your multiple images. In this case I have stacked 4 images vertically

				input = input.to(device, dtype=torch.float)

				labels = labels.to(device, dtype=torch.float)


				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(input)
					# print((outputs.shape,labels.shape))

					loss = calc_loss(outputs, labels, metrics,bce_weight=0.5)



					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				epoch_samples += input.size(0)

			# print_metrics(metrics, epoch_samples, phase)
			epoch_loss = metrics['loss'] / epoch_samples
			epoch_loss_bce = metrics['bce'] / epoch_samples
			epoch_loss_dice = metrics['dice'] / epoch_samples

			if phase == 'val': ## will be useful in next epoch
				to_lr_loss = copy.deepcopy(epoch_loss)
			if (phase == 'train') & (epoch > 0):
				#scheduler.step(to_lr_loss)
				scheduler.step()
				
			# deep copy the model
			if phase == 'val':
				print("saving model")
				if epoch_loss < best_epoch_loss:
					torch.save(model.state_dict(),'bestModel_{ide}_{l}.pth'.format(ide=args['run_id'],l=epoch_loss))
					best_epoch_loss = epoch_loss
					print('Saved bestmodel with best loss {}'.format(best_epoch_loss))
				if (epoch%pth_save_step == 0):
					torch.save(model.state_dict(),'test_modelEpoch_{ep}_{ls}_{ide}.pth'.format(ep =epoch,ls = epoch_loss,ide=args['run_id']))
					print('test_modelEpoch_{ep}_{ls}_{ide}.pth'.format(ep =epoch,ls = epoch_loss,ide=args['run_id']))

			### tensorboard updation

			if phase == 'train':
				writer.add_scalars('{}/trainloss'.format(run_id), {'lr':epoch_lr,'loss':epoch_loss,'bce':epoch_loss_bce,'dice':epoch_loss_dice}, epoch)			
			elif phase == 'val':
				writer.add_scalars('{}/valloss'.format(run_id), {'lr':epoch_lr,'loss':epoch_loss,'bce':epoch_loss_bce,'dice':epoch_loss_dice}, epoch)
				writer.flush()

		time_elapsed = time.time() - since
		print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == "__main__":

	train_set = CircleDataset(image_count=1000,image_size=320,pt_thickness=20,transform=trans,target_transform=target_trans)
	val_set = CircleDataset(image_count=250,image_size=320,pt_thickness=20,transform=trans,target_transform=target_trans)
	print("train val dataset size {},{}".format(len(train_set),len(val_set)))


	# global image_datasets

	image_datasets = {
		'train': train_set, 'val': val_set
	}

	for k,v in image_datasets.items():
		print(k,len(v))
	# exit()


	dataloaders = {
		'train': DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['n_work'], pin_memory=args['pin']),
		'val': DataLoader(val_set, batch_size=args['batch_size'], shuffle=False, num_workers=args['n_work'], pin_memory=args['pin'])
	}



	
	model = ResNetUNet(args['num_class']).to(device)


	# if args['initialization'] == 'xavier':
	# 	model._initialize_()
	# elif args['initialization'] == 'kaiming':
	# 	model._kaiming_initialize_()


	
	if args['optimizer_choice'] == "SGD":
		optimizer_ft =  torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	elif args['optimizer_choice'] == "Adam":
		optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)
	else:
		exit('Wrong optimizer')


	if args['lr_scheduler'] == "ReduceLROnPlateau":
		exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

	elif args['lr_scheduler'] == "CyclicLR":
		exp_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer_ft,base_lr=0.01, max_lr=0.0000000000001, step_size_up=15, step_size_down=15,
			 mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, 
			 max_momentum=0.9, last_epoch=-1, verbose=False) ## base_lr and max_lr are swapped(diff from pytorch documentation, verify if it create a problems)
		if args['verbose']:
			print("base_lr and max_lr are swapped(diff from pytorch documentation, verify if it create a problems)")

	else:
		exit('Wrong lr_scheduler')

	train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=1000, dataloaders=dataloaders)

	writer.close()





