import os
import shutil

import torch
import torchvision
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn

from lion_pytorch import Lion

from pytorch_pretrained_vit import ViT

import time

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

def main():

	epochs = 50

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	transform = transforms.Compose([
		transforms.Resize((384, 384)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	# Load the train and validation datasets
	train_dataset = torchvision.datasets.Places365(root='./data', small=True, split='train-standard', transform=transform)
	val_dataset = torchvision.datasets.Places365(root='./data', small=True, split='val', transform=transform)
	# Create data loaders
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)


	model = ViT('B_16_imagenet1k', pretrained=True)

	num_classes = 365  # Number of classes in Places365
	model.fc = nn.Linear(model.fc.in_features, num_classes)

	model = model.to(device)
	if device == 'cuda':
		model = torch.nn.DataParallel(model)
		cudnn.benchmark = True


	criterion = nn.CrossEntropyLoss()
	optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)


	for epoch in range(epochs):

		#train(train_loader, model, criterion, optimizer, epoch)

		#adjust_learning_rate(optimizer, epoch)

		# train for one epoch
		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()
		top1 = AverageMeter()
		top5 = AverageMeter()

		# switch to train mode
		model.train()

		end = time.time()
		for i, (inputs, targets) in enumerate(train_loader):
			# measure data loading time
			data_time.update(time.time() - end)

			inputs, targets = inputs.to(device), targets.to(device)
			# compute output
			output = model(inputs)
			loss = criterion(output, targets)

			# measure accuracy and record loss
			prec1, prec5 = accuracy(output.data, targets, topk=(1, 5))
			losses.update(loss.data[0], inputs.size(0))
			top1.update(prec1[0], inputs.size(0))
			top5.update(prec5[0], inputs.size(0))

			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % 10 == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
					  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
					   epoch, i, len(train_loader), batch_time=batch_time,
					   data_time=data_time, loss=losses, top1=top1, top5=top5))
		
		# evaluate on validation set

		batch_time = AverageMeter()
		losses = AverageMeter()
		top1 = AverageMeter()
		top5 = AverageMeter()

		# switch to evaluate mode
		model.eval()

		end = time.time()
		for i, (inputs, targets) in enumerate(val_loader):
			inputs, targets = inputs.to(device), targets.to(device)

			# compute output
			output = model(inputs)
			loss = criterion(output, targets)

			# measure accuracy and record loss
			prec1, prec5 = accuracy(output.data, targets, topk=(1, 5))
			losses.update(loss.data[0], inputs.size(0))
			top1.update(prec1[0], inputs.size(0))
			top5.update(prec5[0], inputs.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % 10 == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
					  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
					   i, len(val_loader), batch_time=batch_time, loss=losses,
					   top1=top1, top5=top5))

		print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
			  .format(top1=top1, top5=top5))
		prec1 = top1.avg #validate(val_loader, model, criterion)

		# remember best prec@1 and save checkpoint
		is_best = prec1 > best_prec1
		best_prec1 = max(prec1, best_prec1)
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': "vit",
			'state_dict': model.state_dict(),
			'best_prec1': best_prec1,
		}, is_best, "vit")

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename + '_latest.pth.tar')
	if is_best:
		shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')

if __name__ == '__main__':
	main()