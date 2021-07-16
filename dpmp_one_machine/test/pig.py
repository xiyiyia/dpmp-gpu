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
#        random dataset          
##                    ncll
# 8GPU : 2466.979 samples/sec, total: 20.268 sec/epoch, communication: 0.105 sec/epoch (average)
# 7:     1979.460 samples/sec, total: 25.259 sec/epoch, communication: 0.130 sec/epoch (average)
# 6:     1831.683 samples/sec, total: 27.297 sec/epoch, communication: 0.155 sec/epoch (average)
# 5:     1630.685 samples/sec, total: 30.662 sec/epoch, communication: 0.173 sec/epoch (average)
# 4:     1424.645 samples/sec, total: 35.096 sec/epoch, communication: 0.214 sec/epoch (average)
# 3:     1088.329 samples/sec, total: 45.942 sec/epoch, communication: 0.275 sec/epoch (average)
# 2:     684.848  samples/sec, total: 73.009 sec/epoch, communication: 0.402 sec/epoch (average)
##                  gloo
# 2 :     87.645*2  samples/sec, total: 285.242 sec/epoch, communication: 155.110 sec/epoch (average)
# 3 :     76.204*3 samples/sec, total: 218.702 sec/epoch, communication: 115.256 sec/epoch
# 4 :     66.461*4  samples/sec, total: 188.081 sec/epoch, communication: 95.250 sec/epoch (average)
# 5 :     68.727*5  samples/sec, total: 145.503 sec/epoch, communication: 76.094 sec/epoch (average
# 6 :     54.686*6    samples/sec, total: 152.378 sec/epoch, communication: 77.905 sec/epoch (average)
# 7 :     57.512 samples/sec, total: 124.182 sec/epoch, communication: 64.768 sec/epoch (average)
# 8 :     49.422 samples/sec, total: 126.463 sec/epoch, communication: 62.934 sec/epoch (average)
#         cifar 10  gloo   128/device
# 2 :  602.280 samples/sec, total: 83.018 sec/epoch, communication: 38.650 sec/epoch (average)
# 3 :  502.348 samples/sec, total: 99.533 sec/epoch, communication: 56.495 sec/epoch (average)
# 4 :  429.641 samples/sec, total: 116.376 sec/epoch, communication: 69.092 sec/epoch (average)
# 5 :  1491.902 samples/sec, total: 33.514 sec/epoch, communication: 16.755 sec/epoch (average)
# 6 :  1613.922 samples/sec, total: 30.980 sec/epoch, communication: 15.950 sec/epoch (average)
# 7 :  1806.004 samples/sec, total: 27.685 sec/epoch, communication: 14.772 sec/epoch (average)
# 8 :  1908.463 samples/sec, total: 26.199 sec/epoch, communication: 14.515 sec/epoch (average)

#         cifar 10  gloo   each device has 128 batch size
# 2 :  947.997 samples/sec, total: 52.743 sec/epoch, communication: 20.240 sec/epoch (average)
# 3 :  1187.590 samples/sec, total: 42.102 sec/epoch, communication: 18.747 sec/epoch (average)
# 4 :  1370.870 samples/sec, total: 36.473 sec/epoch, communication: 18.845 sec/epoch (average)
# 5 :  1491.902 samples/sec, total: 33.514 sec/epoch, communication: 16.755 sec/epoch (average)
# 6 :  1613.922 samples/sec, total: 30.980 sec/epoch, communication: 15.950 sec/epoch (average)
# 7 :  1806.004 samples/sec, total: 27.685 sec/epoch, communication: 14.772 sec/epoch (average)
# 8 :  1908.463 samples/sec, total: 26.199 sec/epoch, communication: 14.515 sec/epoch (average)
