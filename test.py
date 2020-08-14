import numpy as np
import matplotlib.pyplot as plt

# plt.axis([0, 10, 0, 1])
y = [1,2,3,4,5,6,7,8,9,10]
i = np.arange(len(y))
plt.scatter(i, y)
# for i, y in zip(i,y):
# 	plt.scatter(i, y)
# for i in range(10):
#     # y = np.random.random()
#     plt.scatter(i, y[i])
#     # plt.pause(0)

plt.show()