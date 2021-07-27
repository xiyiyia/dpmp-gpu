from torchensemble_ import BaggingClassifier  # voting is a classic ensemble strategy
from torchensemble_  import bagging as bagg
from models import resnet,vgg
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import io,sys
import time
theRecodOfLoss0 = bagg.theRecodOfLoss0
theRecodOfLoss1 = bagg.theRecodOfLoss1
theRecodOfLoss2 = bagg.theRecodOfLoss2

def get_Dataloader_model(n,d,batch_size):
  # Load data

  if d == 'cifar10':
    train_transformer = transforms.Compose(
      [
          transforms.ToTensor()
      ]
    )
    test_transformer = transforms.Compose(
      [
          transforms.ToTensor()
      ]
    )
    train_loader = DataLoader(
        datasets.CIFAR10(
            './data', train=True, download=True, transform=train_transformer
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.CIFAR10('./data', train=False, transform=test_transformer),
        batch_size=batch_size,
        shuffle=True,
    )

  if d == 'mnist':
    train_transformer = transforms.Compose(
      [
        #   transforms.RandomHorizontalFlip(),
        #   transforms.RandomCrop(32, 4),
          transforms.ToTensor()
        #   transforms.Normalize(
        #       (0.5), (0.5)
        #   ),
      ]
    )
    test_transformer = transforms.Compose(
      [
          transforms.ToTensor()
        #   transforms.Normalize(
        #       (0.5), (0.5)
        #   ),
      ]
    )
    train_loader = DataLoader(
        datasets.MNIST(
            './data', train=True, download=True, transform=train_transformer
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, transform=test_transformer),
        batch_size=batch_size,
        shuffle=True,
    )
  if d == 'fmnist':
    train_transformer = transforms.Compose(
      [
        #   transforms.RandomHorizontalFlip(),
        #   transforms.RandomCrop(32, 4),
          transforms.ToTensor()
        #   transforms.Normalize(
        #       (0.5), (0.5)
        #   ),
      ]
    )
    test_transformer = transforms.Compose(
      [
          transforms.ToTensor()
        #   transforms.Normalize(
        #       (0.5), (0.5)
        #   ),
      ]
    )
    train_loader = DataLoader(
        datasets.FashionMNIST(
            './data', train=True, download=True, transform=train_transformer
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.FashionMNIST('./data', train=False, transform=test_transformer),
        batch_size=batch_size,
        shuffle=True,
    )

  if d == 'cifar100':
    train_transformer = transforms.Compose(
      [
          transforms.ToTensor()
      ]
    )
    test_transformer = transforms.Compose(
      [
          transforms.ToTensor()
      ]
    )
    train_loader = DataLoader(
        datasets.CIFAR100(
            './data', train=True, download=True, transform=train_transformer
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.CIFAR100('./data', train=False, transform=test_transformer),
        batch_size=batch_size,
        shuffle=True,
    )



    # Define the ensemble
    if n == 'resnet':
      ensemble = BaggingClassifier(
          estimator=resnet.resnet18(num_classes=100),               # here is your deep learning model
          n_estimators=3,                        # number of base estimators
      )
    if n == 'vgg':
      ensemble = BaggingClassifier(
          estimator=vgg.vgg11_bn(num_class=100),               # here is your deep learning model
          n_estimators=3,                        # number of base estimators
      )
    return train_loader,test_loader,ensemble

  # Define the ensemble
  if n == 'resnet':
    ensemble = BaggingClassifier(
        estimator=resnet.resnet18(num_classes=10),               # here is your deep learning model
        n_estimators=3,                        # number of base estimators
    )
  if n == 'vgg':
    ensemble = BaggingClassifier(
        estimator=vgg.vgg11_bn(num_class=10),               # here is your deep learning model
        n_estimators=3,                        # number of base estimators
    )
  return train_loader,test_loader,ensemble
if __name__=="__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-l', type=int, default=0.1, help='learning rate')
  parser.add_argument('-b', type=int, default=128, help='batchsize')
  parser.add_argument('-e', type=int, default=1, help='epoch')
  parser.add_argument('-d', type=str, default='cifar10')
  parser.add_argument('-n', type=str, default='resnet')
  parser.add_argument('-ne', type=int, default=3, help='number of n_estimators')
  args = parser.parse_args()

  train_loader,test_loader,ensemble =  get_Dataloader_model(args.n,args.d,args.b)

  # Set the optimizer
  ensemble.set_optimizer(
      "SGD",                                 # type of parameter optimizer
      lr=args.l,                       # learning rate of parameter optimizer
      weight_decay= 5e-4,              # weight decay of parameter optimizer
      momentum = 0.9
  )

  # Set the learning rate scheduler
  ensemble.set_scheduler(
      "CosineAnnealingLR"   ,                # type of learning rate scheduler
       T_max=args.e                           # additional arguments on the scheduler
  )
  start = time.time()
  # Train the ensemble
  ensemble.fit(
      train_loader,
      epochs=args.e,                          # number of training epochs
  )
  stop = time.time()
  print('time:',stop- start)
  # Evaluate the ensemble
  # acc = ensemble.predict(test_loader)         # testing accuracy

  plt.plot(theRecodOfLoss0, label="estimiter1")
  plt.plot(theRecodOfLoss1, label="estimiter2", linestyle="--")
  plt.plot(theRecodOfLoss2, label="estimiter3", linestyle="-.")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title('Training process')
  plt.legend()  # 打上标签
  plt.show()
  plt.savefig('./pic/'+args.d+args.n+'bagging.png')