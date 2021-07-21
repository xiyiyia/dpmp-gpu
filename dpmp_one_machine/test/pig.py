import matplotlib.pyplot as plt
num_gpu = [2, 3, 4, 5, 6, 7, 8]
# dp_nccl = [43.40,34.77,28.41]
# dp = []
# for i in range(len(num_gpu)):
#   dp.append(num_gpu[i]*num_gpu[i]*num_gpu[i])
dp = [65.6249685, 53.525504, 49.2788225, 41.5384416,40.224256, 38.186636, 37.62017425]#[126.293, 93.709666666666666666666666666667, 82.98975, 77.1052, 72.9215, 70.603428571428571428571428571429, 68.6205] # [319.2651, 245.0382, 200.790675, 192.33954, 180.8925, 177.28045714285714285714285714286, 171.766575]  #gloo
mp = [74.861, 56.722, 43.705, 36.403, 36.000, 35.125, 36.152]#[122.28525, 93.220916666666666666666666666667, 78.2733125, 75.20885, 75.113932291666666666666666666667, 73.856026785714285714285714285714, 74.30419921875] #[250.302, 228.111, 218, 199.368, 179.94, 175.188, 156  ] # chunks = 8
# Plot Average Accuracy vs Communication rounds
plt.figure()
plt.title('vgg19')
# plt.plot(num_gpu, dp, "x-", color='m', label = "data_parrallel_in_nccl")
plt.plot(num_gpu, dp, "+-", color='r', label = "data_parrallel")
plt.plot(num_gpu, mp, "x-", color='m', label = "model_parrallel")
plt.legend()
plt.ylabel('Time(s)')
plt.xlabel('Number of GPUs')
plt.savefig('./time_vgg19.png')


########################################################################################
###########                            VGG11                                   #########
###############        DP:         
##        random dataset             ncll     3x32x32    each method's round is the same  128 391
# 2 :     169602.647 samples/sec, total: 0.295 sec/epoch, communication: 0.007 sec/epoch, training: 0.286 sec/epoch (average)  57.6725
# 3 :     150834.528 samples/sec, total: 0.331 sec/epoch, communication: 0.009 sec/epoch, training: 0.322 sec/epoch (average)  43.1403
# 4 :     124371.560 samples/sec, total: 0.402 sec/epoch, communication: 0.010 sec/epoch, training: 0.391 sec/epoch (average)  39.2955
# 5 :     106933.555 samples/sec, total: 0.468 sec/epoch, communication: 0.016 sec/epoch, training: 0.450 sec/epoch (average)  36.5976
# 6 :     110740.583 samples/sec, total: 0.452 sec/epoch, communication: 0.010 sec/epoch, training: 0.440 sec/epoch (average)  29.455333333333333333333333333333
# 7 :     96367.569 samples/sec, total: 0.519 sec/epoch, communication: 0.009 sec/epoch, training: 0.508 sec/epoch (average)  28.989857142857142857142857142857
# 8 :     81644.800 samples/sec, total: 0.612 sec/epoch, communication: 0.011 sec/epoch, training: 0.600 sec/epoch (average)  29.9115s

##          random dataset   nccl  3x32x32      104 480.769
# 2 :     183348.254 samples/sec, total: 0.273 sec/epoch, communication: 0.011 sec/epoch, training: 0.261 sec/epoch (average) 240.3845  65.6249685
# 3 :     149825.680 samples/sec, total: 0.334 sec/epoch, communication: 0.009 sec/epoch, training: 0.324 sec/epoch (average) 160.256   53.525504
# 4 :     121336.842 samples/sec, total: 0.412 sec/epoch, communication: 0.013 sec/epoch, training: 0.398 sec/epoch (average) 120.19225  49.2788225
# 5 :     115828.787 samples/sec, total: 0.432 sec/epoch, communication: 0.011 sec/epoch, training: 0.420 sec/epoch (average) 96.1538   41.5384416
# 6 :     105852.508 samples/sec, total: 0.502 sec/epoch, communication: 0.009 sec/epoch, training: 0.462 sec/epoch (average) 80.128   40.224256
# 7 :     89897.335 samples/sec, total: 0.556 sec/epoch, communication: 0.009 sec/epoch, training: 0.546 sec/epoch (average) 68.681   38.186636
# 8 :     79819.196 samples/sec, total: 0.626 sec/epoch, communication: 0.008 sec/epoch, training: 0.617 sec/epoch (average)  60.096125  37.62017425
#      [65.6249685, 53.525504, 49.2788225, 41.5384416, ,40.224256, 38.186636, 37.62017425]
#############           MP:
###       random dataset   3x32x32
# 2 :     pipeline-2, 1-1 epochs | 667.905 samples/sec, 74.861 sec/epoch (average)
# 3 :     pipeline-3, 1-1 epochs | 881.485 samples/sec, 56.722 sec/epoch (average)
# 4 :     pipeline-4, 1-1 epochs | 1144.035 samples/sec, 43.705 sec/epoch (average)
# 5 :     pipeline-5, 1-1 epochs | 1373.514 samples/sec, 36.403 sec/epoch (average)
# 6 :     pipeline-6, 1-1 epochs | 1353.027 samples/sec, 36.954 sec/epoch (average)
# 7 :     pipeline-7, 1-1 epochs | 1423.475 samples/sec, 35.125 sec/epoch (average)
# 8 :     pipeline-8, 1-1 epochs | 1383.046 samples/sec, 36.152 sec/epoch (average)
######## [74.861, 56.722, 43.705, 36.403, 36.000, 35.125, 36.152]


#####     unet     random 3x32x32   dp gloo    200 batch     50
### 8 : 8858.021 samples/sec, total: 5.645 sec/epoch, communication: 2.783 sec/epoch, training: 2.860 sec/epoch (average)    141.125
### 7 : 10548.968 samples/sec, total: 4.740 sec/epoch, communication: 2.280 sec/epoch, training: 2.458 sec/epoch (average)  135.4285
### 6 : 11771.407 samples/sec, total: 4.248 sec/epoch, communication: 2.016 sec/epoch, training: 2.230 sec/epoch (average)  141.6
### 5 : 12820.917 samples/sec, total: 3.900 sec/epoch, communication: 1.859 sec/epoch, training: 2.039 sec/epoch (average)  156
### 4 : 14018.280 samples/sec, total: 3.567 sec/epoch, communication: 1.646 sec/epoch, training: 1.919 sec/epoch (average)  178.35
### 3 : 15852.984 samples/sec, total: 3.154 sec/epoch, communication: 1.427 sec/epoch, training: 1.726 sec/epoch (average)  210.26666666666666666666666666667
### 2 : 17949.303 samples/sec, total: 2.786 sec/epoch, communication: 1.244 sec/epoch, training: 1.540 sec/epoch (average)  278



# 41  243.9
### 2 :19095.242 samples/sec, total: 2.618 sec/epoch, communication: 1.246 sec/epoch, training: 1.371 sec/epoch (average)  319.2651
# 3 : 16591.839 samples/sec, total: 3.014 sec/epoch, communication: 1.470 sec/epoch, training: 1.542 sec/epoch (average)   245.0382
# 4 :  15183.599 samples/sec, total: 3.293 sec/epoch, communication: 1.587 sec/epoch, training: 1.705 sec/epoch (average)  200.790675
# 5 :  12680.242 samples/sec, total: 3.943 sec/epoch, communication: 1.858 sec/epoch, training: 2.084 sec/epoch (average)  192.33954
# 6 :  11235.049 samples/sec, total: 4.450 sec/epoch, communication: 2.180 sec/epoch, training: 2.269 sec/epoch (average)  180.8925
# 7 :  9827.667 samples/sec, total: 5.088 sec/epoch, communication: 2.514 sec/epoch, training: 2.573 sec/epoch (average)   177.28045714285714285714285714286
# 8 :  8874.611 samples/sec, total: 5.634 sec/epoch, communication: 2.578 sec/epoch, training: 3.055 sec/epoch (average)   171.766575
# [319.2651, 245.0382, 200.790675, 192.33954, 180.8925, 177.28045714285714285714285714286, 171.766575]
#####unet random 3x32x32 gpipe     39 batch
# 8: pipeline-8, 1-1 epochs | 2456.325 samples/sec, 4.071 sec/epoch (average) 156       256 batch 8 chunks  
### 8: pipeline-8, 1-1 epochs | 2372.964 samples/sec, 4.214 sec/epoch (average) 164.346
#8: pipeline-8, 1-1 epochs | 2113.664 samples/sec, 4.731 sec/epoch (average)

### 7: pipeline-7, 1-1 epochs | 2180.280 samples/sec, 4.587 sec/epoch (average) 178.893
### 7 : pipeline-7, 1-1 epochs | 2491.219 samples/sec, 4.014 sec/epoch (average)
### 7 : pipeline-7, 1-1 epochs | 2187.510 samples/sec, 4.571 sec/epoch (average)
### 7 : pipeline-7, 1-1 epochs | 2226.286 samples/sec, 4.492 sec/epoch (average)  175.188

### 6: pipeline-6, 1-1 epochs | 2390.917 samples/sec, 4.182 sec/epoch (average) 163.098
### 6: pipeline-6, 1-1 epochs | 2194.957 samples/sec, 4.556 sec/epoch (average)
### 6 : pipeline-6, 1-1 epochs | 1847.013 samples/sec, 5.414 sec/epoch (average)
### 6 : pipeline-6, 1-1 epochs | 2167.500 samples/sec, 4.614 sec/epoch (average)  179.94

### 5: pipeline-5, 1-1 epochs | 1956.042 samples/sec, 5.112 sec/epoch (average) 199.368
# 5: pipeline-4, 1-1 epochs | 2224.247 samples/sec, 4.496 sec/epoch (average)

### 4: pipeline-4, 1-1 epochs | 1777.942 samples/sec, 5.624 sec/epoch (average) 218
# 4: pipeline-4, 1-1 epochs | 2246.208 samples/sec, 4.452 sec/epoch (average)
# 4 : pipeline-4, 1-1 epochs | 1714.046 samples/sec, 5.834 sec/epoch (average)

### 3: pipeline-3, 1-1 epochs | 1709.731 samples/sec, 5.849 sec/epoch (average)  228.111
# 3 : pipeline-3, 1-1 epochs | 1503.399 samples/sec, 6.652 sec/epoch (average)

# 2: pipeline-2, 1-1 epochs | 1724.194 samples/sec, 5.800 sec/epoch (average) 226
# 2: pipeline-2, 1-1 epochs | 1899.542 samples/sec, 5.264 sec/epoch (average)
### 2 : pipeline-2, 1-1 epochs | 1558.216 samples/sec, 6.418 sec/epoch (average)  250.302
## [250.302, 228.111, 218, 199.368, 179.94, 175.188, 156  ]



#####     resnet101     random 3x32x32   mp  128 ,  391 batch    
### 8 : pipeline-8, 1-1 epochs | 8213.589 samples/sec, 6.087 sec/epoch (average)  32x128  74.30419921875
### 7 : pipeline-7, 1-1 epochs | 9444.912 samples/sec, 5.294 sec/epoch (average)   73.856026785714285714285714285714
### 6 : pipeline-6, 1-1 epochs | 10833.709 samples/sec, 4.615 sec/epoch (average)  75.113932291666666666666666666667
### 5 : pipeline-5, 1-1 epochs | 12995.558 samples/sec, 3.847 sec/epoch (average)  75.20885
### 4 :  pipeline-4, 1-1 epochs | 15608.921 samples/sec, 3.203 sec/epoch (average) 78.2733125
### 3 :  pipeline-3, 1-1 epochs | 17475.387 samples/sec, 2.861 sec/epoch (average) 93.220916666666666666666666666667
### 2 :  pipeline-2, 1-1 epochs | 19985.877 samples/sec, 2.502 sec/epoch (average) 122.28525
# [122.28525, 93.220916666666666666666666666667, 78.2733125, 75.20885, 75.113932291666666666666666666667, 73.856026785714285714285714285714, 74.30419921875]

###  dp gloo
### 2 ： 77409.563 samples/sec, total: 0.646 sec/epoch, communication: 0.126 sec/epoch, training: 0.495 sec/epoch (average)  126.293
### 3 ： 69513.520 samples/sec, total: 0.719 sec/epoch, communication: 0.126 sec/epoch, training: 0.547 sec/epoch (average) 93.709666666666666666666666666667
### 4 ： 58907.126 samples/sec, total: 0.849 sec/epoch, communication: 0.126 sec/epoch, training: 0.682 sec/epoch (average)  82.98975
### 5 ： 50728.630 samples/sec, total: 0.986 sec/epoch, communication: 0.126 sec/epoch, training: 0.807 sec/epoch (average)  77.1052
### 6 ： 44693.410 samples/sec, total: 1.119 sec/epoch, communication: 0.126 sec/epoch, training: 0.926 sec/epoch (average)  72.9215 ,    6.776
### 7 ： 39544.172 samples/sec, total: 1.264 sec/epoch, communication: 0.126 sec/epoch, training: 1.060 sec/epoch (average)  70.603428571428571428571428571429,  7.246 
### 8 ： 35604.118 samples/sec, total: 1.404 sec/epoch, communication: 0.128 sec/epoch, training: 1.197 sec/epoch (average)  68.6205 ,    7.770

# [126.293, 93.709666666666666666666666666667, 82.98975, 77.1052, 72.9215, 70.603428571428571428571428571429, 68.6205]