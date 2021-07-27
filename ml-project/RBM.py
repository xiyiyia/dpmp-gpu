import sys
import numpy as np
from functions import *
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import io,sys
import matplotlib.pyplot as plt

def get_Dataloader_model(d,batch_size):
    # Load data

    if d == 'cifar10':
        train_transformer = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(28, 4),
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
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(28, 4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5), (0.5)
                ),
            ]
        )
        test_transformer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5), (0.5)
                ),
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
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(28, 4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5), (0.5)
                ),
            ]
        )
        test_transformer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5), (0.5)
                ),
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
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(28, 4),
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

class RBM:
    '''
    设计一个专用于MNIST生成的RBM模型
    '''

    def __init__(self, nv = 784, nh = 128, b = 128, lr = 0.1):
        self.nv = nv
        self.nh = nh
        self.lr = lr
        self.W = np.random.randn(self.nh, self.nv) * 0.1
        self.bv = np.zeros(self.nv)
        self.bh = np.zeros(self.nh)
        self.b = b

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def forword(self, inpt):
        z = np.dot(inpt, self.W.T) + self.bh
        return self.sigmoid(z)

    def backward(self, inpt):
        z = np.dot(inpt, self.W) + self.bv
        return self.sigmoid(z)

    def train_loader(self, X_train):
        np.random.shuffle(X_train)
        self.batches = []
        for i in range(0, len(X_train), self.batch_sz):
            # print(len(X_train),len(X_train[0]))
            self.batches.append(X_train[i:i + self.batch_sz])
        self.indice = 0

    def get_batch(self):
        if self.indice >= len(self.batches):
            return None
        self.indice += 1
        return self.batches[self.indice - 1]

    def fit(self, X_train, epochs=50, batch_sz=128):
        '''
        用梯度上升法做训练
        '''
        self.batch_sz = batch_sz
        err_list = []

        for epoch in range(epochs):
            # 初始化data loader
            self.train_loader(X_train)
            err_sum = 0

            while 1:
                v0_prob = self.get_batch()

                if type(v0_prob) == type(None): break
                size = len(v0_prob)

                dW = np.zeros_like(self.W)
                dbv = np.zeros_like(self.bv)
                dbh = np.zeros_like(self.bh)
                # for v0_prob in  batch_data:
                h0_prob = self.forword(v0_prob)
                h0 = np.zeros_like(h0_prob)
                
                h0[h0_prob > np.random.random(h0_prob.shape)] = 1

                v1_prob = self.backward(h0)
                v1 = np.zeros_like(v1_prob)
                v1[v1_prob > np.random.random(v1_prob.shape)] = 1

                h1_prob = self.forword(v1)
                h1 = np.zeros_like(h1_prob)
                h1[h1_prob > np.random.random(h1_prob.shape)] = 1

                dW = np.dot(h0.T, v0_prob) - np.dot(h1.T, v1_prob)
                dbv = np.sum(v0_prob - v1_prob, axis=0)
                dbh = np.sum(h0_prob - h1_prob, axis=0)

                err_sum += np.mean(np.sum((v0_prob - v1_prob) ** 2, axis=1))

                dW /= size
                dbv /= size
                dbh /= size

                self.W += dW * self.lr
                self.bv += dbv * self.lr
                self.bh += dbh * self.lr

            err_sum = err_sum / len(X_train)
            err_list.append(err_sum)
            print('Epoch {0},err_sum {1}'.format(epoch, err_sum))

        plt.plot(err_list)

    def predict(self, input_x):
        h0_prob = self.forword(input_x)
        h0 = np.zeros_like(h0_prob)
        h0[h0_prob > np.random.random(h0_prob.shape)] = 1
        v1 = self.backward(h0)
        # print(len(v1))
        return v1

def visualize(args,input_x):
    plt.figure(figsize=(5,5), dpi=180)
    for i in range(0,8):
        for j in range(0,8):
            img = np.array(input_x[i*8+j]).reshape(28,28)
            plt.subplot(8,8,i*8+j+1)
            plt.imshow(img ,cmap = plt.cm.gray)
            plt.savefig('./pic/'+args.d+str(i)+str(j)+'.png')
    # plt.figure()
    # plt.title('')
    # #  plt.plot(num_gpu, dp, "x-", color='m', label = "data_parrallel_in_nccl")
    # plt.plot(time_1,loss_1,"+-",color = 'r',label="noise levels: 0")
    # plt.plot(time_2,loss_2,"x-",color = 'b',label="noise levels: 100")
    # plt.plot(time_3,loss_3,"-",color = 'y',label="noise levels: 10000")
    # plt.legend()
    # plt.ylabel('MSE error')
    # plt.xlabel('Transmissions')
    # plt.savefig('./pic/'+args.d+'.png')




def test_rbm(args,k=1):
    train_data, test_data = get_Dataloader_model(args.d,args.b)
    # data = np.array([[]])
    # test = np.array([[]])
    for turn, (batch_x, batch_y) in enumerate(train_data):
        if(len(batch_x) == 128):
            if turn == 0:
                data = batch_x.reshape(128,784).numpy()
            else:
                data = np.append(data,batch_x.reshape(128,784).numpy(),axis=0)
        else:
            break
    for turn, (batch_x, batch_y) in enumerate(test_data):
        if(len(batch_x) == 128):
            if turn == 0:
                test = batch_x.reshape(128,784).numpy()
            else:
                test = np.append(test,batch_x.reshape(128,784).numpy(),axis=0)
        else:
            break
    if(args.d == 'cifar100'):
        rbm = RBM(nh=100,nv=784)
    else:
        rbm = RBM(nh=10,nv=784)
    rbm.fit(data,epochs=args.e)
    visualize(args,rbm.predict(test[0:64]))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=int, default=0.1, help='learning rate')
    parser.add_argument('-b', type=int, default=128, help='batchsize')
    parser.add_argument('-e', type=int, default=100, help='epoch')
    parser.add_argument('-d', type=str, default='mnist')
    parser.add_argument('-n', type=str, default='resnet')
    parser.add_argument('-ne', type=int, default=10, help='number of n_estimators')
    args = parser.parse_args()

    test_rbm(args)
