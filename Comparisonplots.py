import matplotlib.pyplot as plt
import numpy as np

# Edge Probabilities
edge_probabilities = [0.1 * i for i in range(1, 11)]

# Differences
diff_Approx_sim = [-0.008591628769170520, -0.039947588029613300, -0.02787374246568920,
                -0.03089353998985010, -0.012678883445736100, -0.014807937351640300,
                -0.0028148343872685500, -0.0029891646090248500, -0.0014098069069773900,
                5.99439717233652E-07]
diff_Classic_sim = [-0.3985101524218060, -0.2977449186938670, -0.10409060524172100,
                   -0.08971614033470570, -0.029839029419605400, -0.03453587223306480,
                   -0.005778876797704640, -0.006287137929917320, -0.002992282615493370,
                   5.99439717233652E-07]
diff_Markov_sim = [-8.04709363361189E-06, -4.43285779844205E-05, -5.82274829474400E-05,
                5.04419018843849E-05, 3.92502769587383E-05, -4.68719605021331E-05,
                -4.59442497293804E-06, -1.48016122979849E-06, 7.72939944509088E-06,
                5.99439717233760E-07]
diff_new_old = [0.008583581675536910, 0.039903259451628900, 0.02781551498274180,
                0.030943981891734500, 0.012718133722694900, 0.014761065391138200,
                0.002810239962295610, 0.00298768444779505, 0.0014175363064224800,
                1.0842021724855E-19]

# Adjusted figure size
plt.figure(figsize=(4, 3), dpi=200)

# Plotting with improved colors and styles
plt.plot(edge_probabilities, diff_Classic_sim, marker='s',markersize=4, linestyle='--', linewidth=1,
         color='#D55E00', label='Classic - Simulation')
plt.plot(edge_probabilities, diff_Approx_sim, marker='o',markersize=4, linestyle='-', linewidth=1,
         color='#1f77b4', label='Approx - Simulation')
plt.plot(edge_probabilities, diff_Markov_sim, marker='^',markersize=4, linestyle='-.', linewidth=1,
         color='#2ca02c', label='Markov - Simulation')


# Labels and title with smaller fonts
plt.xlabel('Edge Probability', fontsize=8)
plt.ylabel('Difference', fontsize=8)
# Set x-ticks to show all edge probabilities with smaller fonts
plt.xticks(edge_probabilities, [f'{ep:.1f}' for ep in edge_probabilities], fontsize=7)
plt.yticks(fontsize=7)

# Adjust y-axis limits
plt.ylim(min(diff_Classic_sim) * 1.1, max(diff_new_old) * 1.1)

# Legend with smaller font
plt.legend(fontsize=7)

# Grid with thinner lines
plt.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)

# Tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
