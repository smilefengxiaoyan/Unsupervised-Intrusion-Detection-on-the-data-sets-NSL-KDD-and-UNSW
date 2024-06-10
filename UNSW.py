import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# 读取UNSW-NB15数据集
UNSW = pd.read_csv('/Users/smile/Desktop/master paper/master project/UNSW/Training and Testing Sets/UNSW_NB15_training-set.csv')

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

#