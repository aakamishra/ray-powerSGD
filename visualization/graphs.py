import numpy as np
import matplotlib.pyplot as plt
#
#set width of bar
barWidth = 0.25
figure= plt.subplots(figsize =(12, 8))

# set height of bar
# Resnet101
rank1 = [18.22, 12.13, 6.14]
rank2 = [20.62, 13.8, 7.19]
norank = [19.3, 14.25, 5.39]

# Set position of bar on X axis
bx1 = np.arange(len(rank1))
bx2 = [x + barWidth for x in bx1]
bx3 = [x + barWidth for x in bx2]

# Make the plot
plt.bar(bx1, rank1, color ='g', width = barWidth,
        edgecolor ='grey', label ='Rank 1 Compression')
plt.bar(bx2, rank2, color ='b', width = barWidth,
        edgecolor ='grey', label ='Rank 2 Compression')
plt.bar(bx3, norank, color ='r', width = barWidth,
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

#set width of bar
barWidth = 0.25
figure= plt.subplots(figsize =(12, 8))

# Resnet50
rank1 = [22.9, 13.0, 9.6, 5.27]
rank2 = [25.7, 16.0, 11.0, 6.03]
norank = [26.0, 18.0, 11.8, 4.53]

# Set position of bar on X axis
bx1 = np.arange(len(rank1))
bx2 = [x + barWidth for x in bx1]
bx3 = [x + barWidth for x in bx2]

# Make the plot
plt.bar(bx1, rank1, color ='g', width = barWidth,
        edgecolor ='grey', label ='Rank 1 Compression')
plt.bar(bx2, rank2, color ='b', width = barWidth,
        edgecolor ='grey', label ='Rank 2 Compression')
plt.bar(bx3, norank, color ='r', width = barWidth,
        edgecolor ='grey', label ='No Compression')

# Adding Xticks
plt.xlabel('Network Bandwidth', fontsize = 30)
plt.ylabel('Wall time (mins)', fontsize = 30)
plt.yticks(fontsize=20)
plt.xticks([r + barWidth for r in range(len(rank1))],
           ['3 Gbps', '5 Gbps', '10 Gbps', '25 Gbps'], fontsize = 20)
plt.title('ResNet101', fontsize = 30)

plt.legend(fontsize = 20)
plt.show()

#set width of bar
barWidth = 0.25
figure= plt.subplots(figsize =(12, 8))

# set height of bar
# Resnet50 Heterogeneous
dynrank = [12.8, 17.7, 21.0]
norank = [14.53, 19.53, 22.93]
rank3 = [18.40, 24.46, 24.62]

# Set position of bar on X axis
bx1 = np.arange(len(dynrank))
bx2 = [x + barWidth for x in bx1]
bx3 = [x + barWidth for x in bx2]

# Make the plot
plt.bar(bx1, dynrank, color ='g', width = barWidth,
        edgecolor ='grey', label ='Dynamic Compression')
plt.bar(bx2, norank, color ='b', width = barWidth,
        edgecolor ='grey', label ='No Compression')
plt.bar(bx3, rank3, color ='r', width = barWidth,
        edgecolor ='grey', label ='Rank 3 Compression')

# Adding Xticks
# plt.xlabel('Setting', fontsize = 30)
plt.ylabel('Wall time (mins)', fontsize = 30)
plt.yticks(fontsize=20)
plt.xticks([r + barWidth for r in range(len(dynrank))],
           ['Setting 1 (5/5/10/10)', 'Setting 2 (3/5/5/10)', 'Setting 3 (3/3/5/10)'], fontsize = 20)
plt.title('Heterogeneous Network Bandwidth Clusters', fontsize = 30)

plt.legend(fontsize = 20)
plt.show()


# Sample plot
barWidth = 0.25
figure= plt.subplots(figsize =(12, 8))

# set height of bar
Example1 = [400, 200, 350]
Example2 = [280, 150, 250]
Example3 = [320, 175, 280]

# Set position of bar on X axis
bx1 = np.arange(len(Example1))
bx2 = [x + barWidth for x in bx1]
bx3 = [x + barWidth for x in bx2]

# Make the plot
plt.bar(bx1, Example1, color ='r', width = barWidth,
        edgecolor ='grey', label ='No Compression')
plt.bar(bx2, Example2, color ='g', width = barWidth,
        edgecolor ='grey', label ='Rank 1 Compression')
plt.bar(bx3, Example2, color ='b', width = barWidth,
        edgecolor ='grey', label ='Rank 3 Compression')

# Adding Xticks
plt.xlabel('Model Type', fontweight ='bold', fontsize = 15)
plt.ylabel('Wall time (mins) until 50% Accuracy', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(Example1))],
           ['ViT', 'ResNet', 'Third Model'])
plt.title('Wall time for different models in a 2 Gbps network', fontweight ='bold', fontsize = 20)

plt.legend()
plt.show()



