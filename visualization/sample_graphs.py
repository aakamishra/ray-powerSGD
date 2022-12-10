import numpy as np
import matplotlib.pyplot as plt
#
# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

# set height of bar
IT = [65, 45, 30]
ECE = [40, 35, 35]
CSE = [50, 40, 32]

# Set position of bar on X axis
br1 = np.arange(len(IT))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
plt.bar(br1, IT, color ='r', width = barWidth,
        edgecolor ='grey', label ='No Compression')
plt.bar(br2, ECE, color ='g', width = barWidth,
        edgecolor ='grey', label ='Rank 1 Compression')
plt.bar(br3, CSE, color ='b', width = barWidth,
        edgecolor ='grey', label ='Rank 3 Compression')

# Adding Xticks
plt.xlabel('Network Bandwidths', fontweight ='bold', fontsize = 15)
plt.ylabel('Wall time (mins) until 50% Accuracy', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(IT))],
           ['2 Gbps', '10 Gbps', '25 Gbps'])
plt.title('Wall time vs Network Bandwidths in ResNet 50', fontweight ='bold', fontsize = 20)

plt.legend()
plt.show()

# set width of bar
# barWidth = 0.25
# fig = plt.subplots(figsize =(12, 8))
#
# # set height of bar
# IT = [400, 200, 350]
# ECE = [280, 150, 250]
# CSE = [260, 145, 230]
#
# # Set position of bar on X axis
# br1 = np.arange(len(IT))
# br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
#
# # Make the plot
# plt.bar(br1, IT, color ='r', width = barWidth,
#         edgecolor ='grey', label ='No Compression')
# plt.bar(br2, ECE, color ='g', width = barWidth,
#         edgecolor ='grey', label ='Rank 1 Compression')
# plt.bar(br3, CSE, color ='b', width = barWidth,
#         edgecolor ='grey', label ='Dynamic Compression')
#
# # Adding Xticks
# plt.xlabel('Model Type', fontweight ='bold', fontsize = 15)
# plt.ylabel('Wall time (mins) until 50% Accuracy', fontweight ='bold', fontsize = 15)
# plt.xticks([r + barWidth for r in range(len(IT))],
#            ['ViT', 'ResNet', 'Third Model'])
# plt.title('Wall time vs Network Bandwidths in Heterogeneous Clusters', fontweight ='bold', fontsize = 20)
#
# plt.legend()
# plt.show()

# barWidth = 0.25
# fig = plt.subplots(figsize =(12, 8))
#
# # set height of bar
# IT = [400, 200, 350]
# ECE = [280, 150, 250]
# CSE = [320, 175, 280]
#
# # Set position of bar on X axis
# br1 = np.arange(len(IT))
# br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
#
# # Make the plot
# plt.bar(br1, IT, color ='r', width = barWidth,
#         edgecolor ='grey', label ='No Compression')
# plt.bar(br2, ECE, color ='g', width = barWidth,
#         edgecolor ='grey', label ='Rank 1 Compression')
# plt.bar(br3, CSE, color ='b', width = barWidth,
#         edgecolor ='grey', label ='Rank 3 Compression')
#
# # Adding Xticks
# plt.xlabel('Model Type', fontweight ='bold', fontsize = 15)
# plt.ylabel('Wall time (mins) until 50% Accuracy', fontweight ='bold', fontsize = 15)
# plt.xticks([r + barWidth for r in range(len(IT))],
#            ['ViT', 'ResNet', 'Third Model'])
# plt.title('Wall time for different models in a 2 Gbps network', fontweight ='bold', fontsize = 20)
#
# plt.legend()
# plt.show()



