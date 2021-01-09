#!/home/tony/anaconda3/envs/nn_experiments/bin/python

import numpy as np
import logging
import pathlib
import cv2
import os

import matplotlib.pyplot as plt
import copy
import os
import socket
import math
hostname = socket.gethostname()


image_size = 320
class CircleDataset():
	def __init__(self,image_count=None,image_size=None,pt_thickness=None,transform=None, target_transform=None):
		self.ids = range(image_count)

		self.pt_thickness = pt_thickness
		self.image_size = image_size
	 
		self.x1 = [np.random.randint(int(image_size/4),int(image_size/2)) for i in range(image_count)]
		self.y1 = [np.random.randint(int(image_size/4),int(image_size/2)) for i in range(image_count)]
		self.x2 = [np.random.randint(int(image_size/2),int(image_size*0.75)) for i in range(image_count)]
		self.y2 = [np.random.randint(int(image_size/2),int(image_size*0.75)) for i in range(image_count)]
		self.imgs = [np.random.rand(self.image_size,self.image_size,3) for i in range(image_count)]
		self.colors = [(int(np.random.choice([0,256])),int(np.random.choice([0,256])) , int(np.random.choice([0,256]))) for i in range(image_count)]
		



		self.transform = transform
		self.target_transform = target_transform



	def __len__(self,):
		return len(self.ids)


	def __getitem__(self, idx):

		center_pt_x = int((self.x1[idx] + self.x2[idx])/2)
		center_pt_y = int((self.y1[idx] + self.y2[idx])/2)
		r = int(math.sqrt( (center_pt_x - self.x1[idx])**2 + (center_pt_y - self.y1[idx])**2 ))

		# img = np.zeros((self.image_size,self.image_size,3))
		img = self.imgs[idx]
		img = np.clip(img*255,0,255).astype(np.uint8)
		##put first point
		color = self.colors[idx]

		if sum(color) ==0 : # if all are 0
			# print('less color')
			color = (255,0,0)

		# print('color')
		# print(color)
		img = cv2.circle(img, (self.x1[idx],self.y1[idx]), self.pt_thickness, color, -1)
		img = cv2.circle(img, (self.x2[idx],self.y2[idx]), self.pt_thickness, color, -1)

		mask = np.zeros((self.image_size,self.image_size,1))
		mask = cv2.circle(mask, (center_pt_x,center_pt_y), r,(255,255,255), -1)





		img = img/255.0
		mask = mask/255.0

		if self.transform:
			img = self.transform(img)
		if self.target_transform:
			mask = self.target_transform(mask)

		return img, mask


if __name__ == "__main__":

	test_dataset = CircleDataset(image_count=200,image_size=320,pt_thickness=20)


	
	for count, data in enumerate(test_dataset):
		image, mask = data
		print(image.shape)
		print(mask.shape)
		# exit()

		f, axarr = plt.subplots(3)
		axarr[0].imshow(image)
		axarr[0].axis('off')
		# axarr[1,0].imshow(input2[0,:].permute(1,2,0))
		axarr[1].imshow(mask[:,:,0],cmap='gray')
		axarr[1].axis('off')

		axarr[2].imshow(image+mask)
		axarr[2].axis('off')



		# axarr[2].imshow(labels[0,:].permute(1,2,0)[:,:,1],cmap='gray')
		# axarr[3].imshow(labels[0,:].permute(1,2,0)[:,:,2],cmap='gray')
		# axarr.axis('off')
		plt.savefig('img.jpg', dpi=800)

		plt.show()
		# plt.pause(0.08)
		# plt.close()
