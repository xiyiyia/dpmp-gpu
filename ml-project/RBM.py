import sys
import numpy
from functions import *
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
    return train_loader,test_loader

class RBM :
    def __init__(self, input=None, n_visible=2, n_hidden=3, W=None, hbias=None, vbias=None, rng=None):
        
        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer

        if rng is None:
            rng = numpy.random.RandomState(1234)


        if W is None:
            a = 1. / n_visible
            initial_W = numpy.array(rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

            W = initial_W

        if hbias is None:
            hbias = numpy.zeros(n_hidden)  # initialize h bias 0

        if vbias is None:
            vbias = numpy.zeros(n_visible)  # initialize v bias 0


        self.rng = rng
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias


    def contrastive_divergence(self, lr=0.1, k=1, input=None):
        if input is not None:
            self.input = input
        
        ''' CD-k '''
        ph_mean, ph_sample = self.sample_h_given_v(self.input)

        chain_start = ph_sample

        for step in xrange(k):
            if step == 0:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(chain_start)
            else:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(nh_samples)

        # chain_end = nv_samples


        self.W += lr * (numpy.dot(self.input.T, ph_mean) - numpy.dot(nv_samples.T, nh_means))
        self.vbias += lr * numpy.mean(self.input - nv_samples, axis=0)
        self.hbias += lr * numpy.mean(ph_mean - nh_means, axis=0)


    def sample_h_given_v(self, v0_sample):
        h1_mean = self.propup(v0_sample)
        h1_sample = self.rng.binomial(size=h1_mean.shape,   # discrete: binomial
                                       n=1,
                                       p=h1_mean)

        return [h1_mean, h1_sample]


    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.rng.binomial(size=v1_mean.shape,   # discrete: binomial
                                            n=1,
                                            p=v1_mean)
        
        return [v1_mean, v1_sample]

    def propup(self, v):
        pre_sigmoid_activation = numpy.dot(v, self.W) + self.hbias
        return sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        pre_sigmoid_activation = numpy.dot(h, self.W.T) + self.vbias
        return sigmoid(pre_sigmoid_activation)


    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return [v1_mean, v1_sample,
                h1_mean, h1_sample]
    

    def get_reconstruction_cross_entropy(self):
        pre_sigmoid_activation_h = numpy.dot(self.input, self.W) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)
        
        pre_sigmoid_activation_v = numpy.dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)

        cross_entropy =  - numpy.mean(
            numpy.sum(self.input * numpy.log(sigmoid_activation_v) +
            (1 - self.input) * numpy.log(1 - sigmoid_activation_v),
                      axis=1))
        
        return cross_entropy

    def reconstruct(self, v):
        h = sigmoid(numpy.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(numpy.dot(h, self.W.T) + self.vbias)
        return reconstructed_v





def test_rbm(learning_rate=0.1, k=1, training_epochs=1000):
    
    
    train_data, test_data = get_Dataloader_model()
    print(train_data)

    rng = numpy.random.RandomState(123)

    # construct RBM
    rbm = RBM(input=data, n_visible=6, n_hidden=2, rng=rng)

    # train
    for epoch in xrange(training_epochs):
        rbm.contrastive_divergence(lr=learning_rate, k=k)


    # test
    v = numpy.array([[1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0]])

    # print rbm.reconstruct(v)



if __name__ == "__main__":
    test_rbm()