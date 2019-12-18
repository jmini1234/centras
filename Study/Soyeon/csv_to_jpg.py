import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('C:/Users/CSE_125-2/Desktop/output.csv', sep=';',header=None)
pixels = np.array(df).reshape(240,240,3)

plt.imshow(pixels)
plt.show()
