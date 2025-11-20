import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter

# 1. 加载清洗后的数据（需确保文件路径正确）
data_dir = "./"  # 替换为实际数据路径
train_df = pd.read_csv(f"{data_dir}/train_cleaned.csv")
valid_df = pd.read_csv(f"{data_dir}/valid_cleaned.csv")
test_df = pd.read_csv(f"{data_dir}/test_cleaned.csv")

# 2. 统计标签分布
def analyze_label_distribution(df, name):
    print(f"\n===== {name} 标签分布统计 ======")
    counts = df["label"].value_counts()
    print(counts)
    return counts.to_dict()  # 返回字典格式结果

train_counts = analyze_label_distribution(train_df, "训练集")
valid_counts = analyze_label_distribution(valid_df, "验证集")
test_counts = analyze_label_distribution(test_df, "测试集")

# 3. 可视化标签分布
sns.set(style="whitegrid", font_scale=1.2)

def plot_distribution(df, name):
    plt.figure(figsize=(8, 5))
    sns.countplot(x="label", data=df, palette="Set2")
    plt.title(f"{name} 标签分布", fontsize=14)
    plt.xlabel("情感标签（0=负向，1=正向）", fontsize=12)
    plt.ylabel("样本数量", fontsize=12)
    plt.xticks([0, 1], ["负向", "正向"])  # 自定义标签显示
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{name}_label_distribution.png")  # 保存图片
    plt.show()

plot_distribution(train_df, "训练集")
plot_distribution(valid_df, "验证集")
plot_distribution(test_df, "测试集")

# 4. 保存分析结果到JSON文件
label_distribution = {
    "train": train_counts,
    "valid": valid_counts,
    "test": test_counts
}

with open("label_distribution.json", "w") as f:
    json.dump(label_distribution, f, indent=4)

print("\n标签分布分析完成！结果已保存到 label_distribution.json")