import numpy as np
import matplotlib.pyplot as plt
#
# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

# set height of bar
# Resnet101
rank1 = [18.22, 12.13, 6.14]
rank2 = [20.62, 13.8, 7.19]
norank = [19.3, 14.25, 5.39]

# Resnet50
# rank1 = [22.9, 13.0, 9.6, 5.27]
# rank2 = [25.7, 16.0, 11.0, 6.03]
# norank = [26.0, 18.0, 11.8, 4.53]

# Set position of bar on X axis
br1 = np.arange(len(rank1))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
plt.bar(br1, rank1, color ='g', width = barWidth,
        edgecolor ='grey', label ='Rank 1 Compression')
plt.bar(br2, rank2, color ='b', width = barWidth,
        edgecolor ='grey', label ='Rank 2 Compression')
plt.bar(br3, norank, color ='r', width = barWidth,
        edgecolor ='grey', label ='No Compression')

# Adding Xticks
plt.xlabel('Network Bandwidth', fontsize = 30)
plt.ylabel('Wall time (mins)', fontsize = 30)
plt.yticks(fontsize=20)
plt.xticks([r + barWidth for r in range(len(rank1))],
           ['5 Gbps', '10 Gbps', '25 Gbps'], fontsize = 20)
plt.title('ResNet101', fontsize = 30)

plt.legend(fontsize = 20)
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



