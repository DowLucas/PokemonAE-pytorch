import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

losses = np.load("loss-1578837142.npy")

plt.plot([i for i in range(len(losses))], losses)
plt.show()

