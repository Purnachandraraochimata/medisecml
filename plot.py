import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
data = pd.read_csv("results/metrics.csv")

# PSNR Graph
plt.plot(data["PSNR"])
plt.title("PSNR Values")
plt.xlabel("Images")
plt.ylabel("PSNR")
plt.show()

# Entropy Graph
plt.plot(data["Entropy"])
plt.title("Entropy Values")
plt.xlabel("Images")
plt.ylabel("Entropy")
plt.show()