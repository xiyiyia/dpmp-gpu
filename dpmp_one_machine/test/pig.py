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
#         cifar 10  gloo
# 2 :  806.687 samples/sec, total: 61.982 sec/epoch, communication: 19.195 sec/epoch (average)
# 3 :  811.002 samples/sec, total: 61.652 sec/epoch, communication: 19.387 sec/epoch (average)
# 4 :  831.916 samples/sec, total: 60.102 sec/epoch, communication: 17.298 sec/epoch (average)
# 5 :  813.038 samples/sec, total: 61.498 sec/epoch, communication: 16.533 sec/epoch (average)
# 6 :  714.844 samples/sec, total: 69.945 sec/epoch, communication: 16.370 sec/epoch (average)
# 7 :  736.280 samples/sec, total: 67.909 sec/epoch, communication: 14.787 sec/epoch (average)
# 8 :  632.884 samples/sec, total: 79.003 sec/epoch, communication: 18.466 sec/epoch (average)
