# Improved function to call phenotypes based on info

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#def better_phenotype(data):


data = pd.read_csv(r"Z:\users\6331823\Mario Pipeline\frames-frames_extracted\Second try\cellpose\Channel-ref-frame-0_quantified.csv")


data["avgWT"] = data["Raw_WT"] / data["Volume"]
data["avgCRC"] = data["Raw_CRC"] / data["Volume"]
data["ratioWTCRC"] = data["avgWT"] / data["avgCRC"]

print((data["ratioWTCRC"] > 0.5).sum())
print(data[data["ratioWTCRC"] > 0.5])


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(data.sort_values(by="avgWT", ascending=False))


ratios = data["avgWT"]
ratios = ratios[ratios > 0]

#bins = np.logspace(np.log10(ratios.min()), np.log10(ratios.max()), 30)

plt.hist(data["avgWT"], bins=30, color='skyblue', edgecolor='black')
plt.xlabel("WT/CRC Ratio")
plt.ylabel("Frequency")

#plt.xscale('log')
plt.title("Histogram of WT/CRC Ratio")
plt.show()