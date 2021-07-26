from torchensemble_ import VotingClassifier  # voting is a classic ensemble strategy
from models import resnet,vgg
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import io,sys

def get_Dataloader_model(n,d,batch_size):
  # Load data
  train_transformer = transforms.Compose(
      [
          transforms.RandomHorizontalFlip(),
          transforms.RandomCrop(32, 4),
          transforms.ToTensor(),
          transforms.Normalize(
              (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
          ),
      ]
  )
  test_transformer = transforms.Compose(
      [
          transforms.ToTensor(),
          transforms.Normalize(
              (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
          ),
      ]
  )

  if d == 'cifar10':
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
      ensemble = VotingClassifier(
          estimator=resnet.resnet18(num_classes=100),               # here is your deep learning model
          n_estimators=10,                        # number of base estimators
      )
    if n == 'vgg':
      ensemble = VotingClassifier(
          estimator=vgg.vgg11_bn(num_classes=100),               # here is your deep learning model
          n_estimators=10,                        # number of base estimators
      )
    return train_loader,test_loader,ensemble
  # Define the ensemble
  if n == 'resnet':
    ensemble = VotingClassifier(
        estimator=resnet.resnet18(num_classes=10),               # here is your deep learning model
        n_estimators=10,                        # number of base estimators
    )
  if n == 'vgg':
    ensemble = VotingClassifier(
        estimator=vgg.vgg11_bn(num_classes=10),               # here is your deep learning model
        n_estimators=10,                        # number of base estimators
    )
  return train_loader,test_loader,ensemble
if __name__=="__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-l', type=int, default=0.1, help='learning rate')
  parser.add_argument('-b', type=int, default=128, help='batchsize')
  parser.add_argument('-e', type=int, default=100, help='epoch')
  parser.add_argument('-d', type=str, default='cifar10')
  parser.add_argument('-n', type=str, default='resnet')
  parser.add_argument('-ne', type=int, default=10, help='number of n_estimators')
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

  # Train the ensemble
  ensemble.fit(
      train_loader,
      epochs=args.e,                          # number of training epochs
  )

  # Evaluate the ensemble
  acc = ensemble.predict(test_loader)         # testing accuracy
