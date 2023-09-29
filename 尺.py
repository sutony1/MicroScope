import cv2
import numpy as np
import matplotlib.pyplot as plt
imgPath="D:\\rule\\rule_black.jpg"#D:\\rule1.png
img = cv2.imread(imgPath)  # vd.jpg,rice1dilate.jpg "D:\\blood_bao1s.jpg"
mean_img=cv2.pyrMeanShiftFiltering(img,20,30)#数值设置得大，就慢一些,20是半径，30是色彩范围
gray_img = cv2.cvtColor(mean_img, cv2.COLOR_BGR2GRAY)

th2=cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)#11比5好,mean比高斯好，mean封闭性好，如果是白尺黑刻度，用Thresh_Binary_inv
from scipy.signal import argrelextrema

# 投影白点到x轴上
projection = np.sum(th2, axis=0)
# 计算一阶导数

gradient = np.gradient(projection)

# 找到投影结果中的局部最大值，即峰值的位置
peaks = argrelextrema(projection, np.greater)

# 计算峰值之间的距离
distances = np.diff(peaks)

# 打印白点数随x轴的分布和峰值之间的距离
print("白点数随x轴的分布:", projection)
print("峰值的位置:", peaks)
print("峰值之间的距离:", distances)
from scipy.signal import find_peaks
from sklearn.cluster import KMeans


# 找出所有的峰值
peaks, _ = find_peaks(projection)

# 获取峰值的高度
peak_heights = projection[peaks]

# 使用K-Means聚类算法将峰值分组
kmeans = KMeans(n_clusters=2, random_state=0).fit(peak_heights.reshape(-1, 1))

# 找出高的峰值和低的峰值
high_peaks = peaks[kmeans.labels_ == 0]  # 假设高的峰值在第0个聚类中
low_peaks = peaks[kmeans.labels_ == 1]  # 假设低的峰值在第1个聚类中

print("高的峰值:", high_peaks)
print("低的峰值:", low_peaks)

# 使用matplotlib绘制白点数随x轴的分布和一阶导数曲线
plt.figure()
plt.plot(projection, label='Projection')
plt.scatter(high_peaks, projection[high_peaks], color='blue', label='High peaks')  # 绘制高的峰值
plt.scatter(low_peaks, projection[low_peaks], color='red', label='Low peaks')  # 绘制低的峰值
plt.title('Distribution of white points along x-axis')
plt.xlabel('x')
plt.ylabel('Number of white points')
plt.legend()
plt.show()
cv2.imshow('Detected Lines', th2)
cv2.waitKey()


