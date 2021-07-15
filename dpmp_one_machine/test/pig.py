import matplotlib.pyplot as plt
num_gpu = [2, 4, 8]
# dp_nccl = [43.40,34.77,28.41]
dp = [104.15724873542786,85.22297787666321,78.66554164886475, 75.7890260219574, 66.05054545402527, 68.49734902381897,66.56337547302246]  #nccl
mp = [78.02846360206604, 76.4333119392395] # chunks = 8
# Plot Average Accuracy vs Communication rounds
plt.figure()
plt.title('batchsize=640, chunks=6')
# plt.plot(num_gpu, dp_nccl, "x-", color='m', label = "data_parrallel_in_nccl")
plt.plot(num_gpu, dp, "+-", color='r', label = "data_parrallel")
plt.plot(num_gpu, mp, "x-", color='m', label = "model_parrallel")
plt.legend()
plt.ylabel('time(s)')
plt.xlabel('number of gpus')
plt.savefig('./time.png')