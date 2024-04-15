import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# 读取UNSW-NB15数据集
UNSW = pd.read_csv('/Users/smile/Desktop/maste paper/python project/UNSW/Training and Testing Sets/UNSW_NB15_training-set.csv')

# 去重处理，获取唯一的攻击类型
unique_attack_types = UNSW['attack_cat'].unique()

# 统计每种攻击类型的出现次数
attack_counts = UNSW['attack_cat'].value_counts()

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.bar(attack_counts.index, attack_counts.values, color='skyblue')
plt.xlabel('Attack Category')
plt.ylabel('Count')
plt.title('Distribution of Attack Categories')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



UNSW_Attack_type = UNSW['attack_cat']

sns.countplot(x='attack_cat',  data=UNSW_Attack_type)

# 查看数据集的前几行
print(data.head())

# 查看数据集的统计摘要
print(data.describe())

# 查看数据集的列名
print(data.columns)

# 检查缺失值
print(data.isnull().sum())

# 查看数据集中不同类别的计数
print(data['label'].value_counts())



# 查看不同类别的计数并绘制条形图
plt.figure(figsize=(8, 6))
data['label'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Counts of Different Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 绘制相关性矩阵的热力图
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title('Correlation Matrix Heatmap')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.show()

# 进行其他可能的可视化和分析...

