# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

x = np.array([1, 2, 3, 4])
label_1 = np.array([66.89,69.62,70.19,70.74])
label_3 = np.array([74.35,75.59,75.96,76.23])
label_10 = np.array([76.66,77.48,77.27,76.92])

# label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
# color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
# 线型：-  --   -.  :    ,
# marker：.  ,   o   v    <    *    +    1
plt.figure(figsize=(8, 5))
plt.grid(linestyle=":")  # 设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框


plt.plot(x, label_1, marker='o',markersize = 8, color="blue", label="Label #1", linewidth=1,linestyle="--")
plt.plot(x, label_3, marker='s', color="green",markersize = 8, label="Label #3", linewidth=1,linestyle="--")
plt.plot(x, label_10, marker='*', color="red",markersize = 9, label="Label #10", linewidth=1,linestyle="--")

group_labels = ['Teacher', 'Re-train #1', 'Re-train #2', 'Re-train #3']  # x轴刻度的标识
plt.xticks(x, group_labels, fontsize=16, fontweight='normal')  # 默认字体大小为10
plt.yticks(fontsize=16, fontweight='normal')
# plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
plt.xlabel("Iteration number of re-train ", fontsize=16, fontweight='normal')
plt.ylabel("F1 (%)", fontsize=16, fontweight='normal')
plt.xlim(0.9, 4.1)  # 设置x轴的范围
plt.ylim(65.5, 78.5)

# plt.legend()          #显示各曲线的图例
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=16, fontweight='normal')  # 设置图例字体的大小和粗细

plt.savefig('./ssl.pdf', format='pdf')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
plt.show()
