# Import libraries
import matplotlib.pyplot as plt
import numpy as np

# Creating dataset
np.random.seed(79)

data_1 = np.random.uniform(45, 98, 500)
data_2 = np.random.uniform(47, 100, 500)
data_3 = np.random.uniform(49, 102, 500)
data_4 = np.random.uniform(51, 104, 500)
data_5 = np.random.uniform(53, 109, 500)

data = [data_1, data_2, data_3, data_4, data_5]

fig = plt.figure(figsize =(2, 4))

#set font size using rcParams
plt.rcParams.update({'font.size': 8})


ax = fig.add_subplot(111)

# Creating axes instance
bp = ax.boxplot(data, patch_artist = True,
                notch = False, vert = 1, showfliers = False)
colors = ['grey', 'grey', 'grey', 'grey', 'grey']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# changing color and linewidth of
# whiskers
for whisker in bp['whiskers']:
    whisker.set(color ='black',
                linewidth = 1,
                linestyle ="-")

# changing color and linewidth of
# caps
for cap in bp['caps']:
    cap.set(color ='black',
            linewidth = 1)

# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color ='black',
               linewidth = 1)

# changing style of fliers
# for flier in bp['fliers']:
#     flier.set(marker ='D',
#               color ='#e7298a',
#               alpha = 0.5)
    
# x-axis labels
ax.set_xticklabels(['0.08', '0.09', '0.1', '0.11', '0.12'])

#set y-axis label
# ax.set_ylabel('space used ($km^2$)')

# Adding title 
# plt.title("Customized box plot")

# Removing top axes and right axes ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
    
plt.savefig('sensitivity_analysis/plots/forest-food-percent/codes/box_plots_MCP/S1_agg_0.8_month_dry.png', dpi = 300, bbox_inches = 'tight')