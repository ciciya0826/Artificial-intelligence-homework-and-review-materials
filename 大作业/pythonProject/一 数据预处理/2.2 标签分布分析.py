import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 加载清洗后的数据
train_df = pd.read_csv('train_cleaned.csv')
valid_df = pd.read_csv('valid_cleaned.csv')
test_df = pd.read_csv('test_cleaned.csv')

# 分析标签分布
print("\n训练集标签分布:")
print(train_df['label'].value_counts())
print("\n验证集标签分布:")
print(valid_df['label'].value_counts())
print("\n测试集标签分布:")
print(test_df['label'].value_counts())

# 可视化标签分布
sns.set(style="whitegrid")

def plot_label_distribution(df, title):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df, palette='Set2')
    plt.title(title)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.close()

print("\n正在生成标签分布可视化图表...")
plot_label_distribution(train_df, '训练集标签分布')
plot_label_distribution(valid_df, '验证集标签分布')
plot_label_distribution(test_df, '测试集标签分布')

# 保存标签分布分析结果
label_distribution = {
    'train': train_df['label'].value_counts().to_dict(),
    'valid': valid_df['label'].value_counts().to_dict(),
    'test': test_df['label'].value_counts().to_dict()
}

with open('label_distribution.json', 'w') as f:
    json.dump(label_distribution, f, indent=4)