import pandas as pd

# 加载数据集
train_df = pd.read_csv('../Train.csv')
valid_df = pd.read_csv('../Valid.csv')
test_df = pd.read_csv('../Test.csv')

# 查看数据的前几行
print(train_df.head())
print(valid_df.head())
print(test_df.head())

# 检查数据的基本信息
print(train_df.info())
print(valid_df.info())
print(test_df.info())