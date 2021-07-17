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
##        random dataset             ncll         each method's round is the same
# 2 :     649.560 samples/sec, total: 76.975 sec/epoch, communication: 0.791 sec/epoch (average)
# 3 :     874.324 samples/sec, total: 57.187 sec/epoch, communication: 0.792 sec/epoch (average)
# 4 :     1171.265 samples/sec, total: 42.689 sec/epoch, communication: 0.812 sec/epoch (average)
# 5 :     1083.847 samples/sec, total: 46.132 sec/epoch, communication: 0.875 sec/epoch (average)
# 6 :     1146.889 samples/sec, total: 43.596 sec/epoch, communication: 0.934 sec/epoch (average)
# 7 :     1306.463 samples/sec, total: 38.271 sec/epoch, communication: 0.940 sec/epoch (average)
# 8 :     1654.698 samples/sec, total: 30.217 sec/epoch, communication: 0.950 sec/epoch (average)

##          random dataset 3x224x224        gloo  round == 50
# 2 :     308.115 samples/sec, total: 162.277 sec/epoch, communication: 38.000 sec/epoch, training: 124.254 sec/epoch (average)
# 3 :     380.280 samples/sec, total: 131.482 sec/epoch, communication: 42.000 sec/epoch, training: 89.987 sec/epoch (average)
# 4 :     402.852 samples/sec, total: 124.115 sec/epoch, communication: 46.000 sec/epoch, training: 78.053 sec/epoch (average)
# 5 :     383.023 samples/sec, total: 130.540 sec/epoch, communication: 55.000 sec/epoch, training: 75.775 sec/epoch (average)
# 6 :     375.396 samples/sec, total: 133.193 sec/epoch, communication: 0.000 sec/epoch, training: 72.345 sec/epoch (average)
# 7 :     368.508 samples/sec, total: 135.682 sec/epoch, communication: 0.000 sec/epoch, training: 68.240 sec/epoch (average)
# 8 :     385.432 samples/sec, total: 129.725 sec/epoch, communication: 0.000 sec/epoch, training: 63.924 sec/epoch (average)

#############           MP:
###       random dataset   3x224x224 
# 2 :     pipeline-2, 1-1 epochs | 371.910 samples/sec, 134.441 sec/epoch (average)
# 3 :     pipeline-3, 1-1 epochs | 495.354 samples/sec, 100.938 sec/epoch (average)
# 4 :     pipeline-4, 1-1 epochs | 456.225 samples/sec, 109.595 sec/epoch (average)
# 5 :     pipeline-5, 1-1 epochs | 457.693 samples/sec, 109.243 sec/epoch (average)
# 6 :     pipeline-6, 1-1 epochs | 703.784 samples/sec, 71.045 sec/epoch (average)
# 7 :     pipeline-7, 1-1 epochs | 933.117 samples/sec, 53.584 sec/epoch (average)
# 8 :     pipeline-8, 1-1 epochs | 543.907 samples/sec, 91.927 sec/epoch (average)

##          random dataset 3x224x224  128/devices      gloo
# 2 :     205.028 samples/sec, total: 243.870 sec/epoch, communication: 139.224 sec/epoch (average)
# 3 :     261.787 samples/sec, total: 190.995 sec/epoch, communication: 106.195 sec/epoch (average)
# 4 :     325.373 samples/sec, total: 153.670 sec/epoch, communication: 84.616 sec/epoch (average)
# 5 :     310.993 samples/sec, total: 160.775 sec/epoch, communication: 88.187 sec/epoch
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


#####     unet     random 3x32x32
### 8 : 496.102 samples/sec, total: 100.786 sec/epoch, communication: 2.999 sec/epoch, training: 85.073 sec/epoch (average)
