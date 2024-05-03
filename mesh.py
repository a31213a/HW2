import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in range(-10,10,1):
    for j in range(-10,10,1):
        x=i
        y=j
        z=-2*(x-2)**2-3*(y+3)**2
        ax.scatter(x, y, z, marker='o')
        print(i,j)
# 顯示圖例
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()