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
##                    ncll
# 8GPU : 2466.979/8 samples/sec, total: 20.268 sec/epoch, communication: 0.105*8 sec/epoch (average)
# 7:     1979.460/7 samples/sec, total: 25.259 sec/epoch, communication: 0.130*7 sec/epoch (average)
# 6:     1831.683/6 samples/sec, total: 27.297 sec/epoch, communication: 0.155*6 sec/epoch (average)
# 5:     1630.685/5 samples/sec, total: 30.662 sec/epoch, communication: 0.173*5 sec/epoch (average)
# 4:     1424.645/4 samples/sec, total: 35.096 sec/epoch, communication: 0.214*4 sec/epoch (average)
# 3:     1088.329/3 samples/sec, total: 45.942 sec/epoch, communication: 0.275*3 sec/epoch (average)
# 2:     684.848 samples/sec, total: 73.009 sec/epoch, communication: 0.402*2 sec/epoch (average)