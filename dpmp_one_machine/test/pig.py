import matplotlib.pyplot as plt
num_gpu = [2, 4, 8]
# dp_nccl = [43.40,34.77,28.41]
dp = [131,99,73]  #gloo
mp = [143.932,88,85] # chunks = 8
# Plot Average Accuracy vs Communication rounds
plt.figure()
plt.title('vgg11,class=1000')
# plt.plot(num_gpu, dp_nccl, "x-", color='m', label = "data_parrallel_in_nccl")
plt.plot(num_gpu, dp, "+-", color='r', label = "data_parrallel")
plt.plot(num_gpu, mp, "x-", color='m', label = "model_parrallel")
plt.legend()
plt.ylabel('time(s)')
plt.xlabel('number of gpus')
plt.savefig('./time.png')


########################################################################################
###########                            VGG11                                   #########
###############        DP:         
##        random dataset             ncll
# 8GPU : 2466.979 samples/sec, total: 20.268 sec/epoch, communication: 0.105 sec/epoch (average)
# 7:     1979.460 samples/sec, total: 25.259 sec/epoch, communication: 0.130 sec/epoch (average)
# 6:     1831.683 samples/sec, total: 27.297 sec/epoch, communication: 0.155 sec/epoch (average)
# 5:     1630.685 samples/sec, total: 30.662 sec/epoch, communication: 0.173 sec/epoch (average)
# 4:     1424.645 samples/sec, total: 35.096 sec/epoch, communication: 0.214 sec/epoch (average)
# 3:     1088.329 samples/sec, total: 45.942 sec/epoch, communication: 0.275 sec/epoch (average)
# 2:     684.848  samples/sec, total: 73.009 sec/epoch, communication: 0.402 sec/epoch (average)

##          random dataset 3x224x224        gloo
# 2 :     87.645*2  samples/sec, total: 285.242 sec/epoch, communication: 155.110 sec/epoch (average)
# 3 :     76.204*3 samples/sec, total: 218.702 sec/epoch, communication: 115.256 sec/epoch
# 4 :     66.461*4  samples/sec, total: 188.081 sec/epoch, communication: 95.250 sec/epoch (average)
# 5 :     68.727*5  samples/sec, total: 145.503 sec/epoch, communication: 76.094 sec/epoch (average
# 6 :     54.686*6    samples/sec, total: 152.378 sec/epoch, communication: 77.905 sec/epoch (average)
# 7 :     57.512 samples/sec, total: 124.182 sec/epoch, communication: 64.768 sec/epoch (average)
# 8 :     49.422 samples/sec, total: 126.463 sec/epoch, communication: 62.934 sec/epoch (average)

##          random dataset 3x224x224  128/devices      gloo
# 2 :     183.789 samples/sec, total: 272.051 sec/epoch, communication: 148.089 sec/epoch (average)
# 3 :     261.211 samples/sec, total: 191.416 sec/epoch, communication: 105.979 sec/epoch (average)
# 4 :     325.373 samples/sec, total: 153.670 sec/epoch, communication: 84.616 sec/epoch (average)
# 5 :     68.727*5  samples/sec, total: 145.503 sec/epoch, communication: 76.094 sec/epoch (average
# 6 :     54.686*6    samples/sec, total: 152.378 sec/epoch, communication: 77.905 sec/epoch (average)
# 7 :     57.512 samples/sec, total: 124.182 sec/epoch, communication: 64.768 sec/epoch (average)
# 8 :     49.422 samples/sec, total: 126.463 sec/epoch, communication: 62.934 sec/epoch (average)

##                                   resnet18
##         cifar 10  gloo   128/device
# 2 :  602.280 samples/sec, total: 83.018 sec/epoch, communication: 38.650 sec/epoch (average)
# 3 :  502.348 samples/sec, total: 99.533 sec/epoch, communication: 56.495 sec/epoch (average)
# 4 :  429.641 samples/sec, total: 116.376 sec/epoch, communication: 69.092 sec/epoch (average)
# 5 :  409.107 samples/sec, total: 122.218 sec/epoch, communication: 79.600 sec/epoch (average)
# 6 :  360.694 samples/sec, total: 138.622 sec/epoch, communication: 91.900 sec/epoch (average)
# 7 :  336.272 samples/sec, total: 148.689 sec/epoch, communication: 101.123 sec/epoch (average)
# 8 :  311.115 samples/sec, total: 160.712 sec/epoch, communication: 109.890 sec/epoch (average)

#         random  3x32x32  gloo   
# 2 :  1025.661 samples/sec, total: 48.749 sec/epoch, communication: 21.684 sec/epoch (average)
# 3 :  1373.801 samples/sec, total: 36.395 sec/epoch, communication: 17.953 sec/epoch (average)
# 4 :  1396.486 samples/sec, total: 35.804 sec/epoch, communication: 18.470 sec/epoch (average)
# 5 :  1589.967 samples/sec, total: 31.447 sec/epoch, communication: 16.428 sec/epoch (average)
# 6 :  1612.644 samples/sec, total: 31.005 sec/epoch, communication: 16.224 sec/epoch (average)
# 7 :  1853.545 samples/sec, total: 26.975 sec/epoch, communication: 15.061 sec/epoch (average)
# 8 :  1886.617 samples/sec, total: 26.502 sec/epoch, communication: 14.867 sec/epoch (average)
