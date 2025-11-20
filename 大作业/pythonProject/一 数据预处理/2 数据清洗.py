import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

# 下载必要的nltk数据
nltk.download('stopwords')
nltk.download('wordnet')

# 定义文本清洗函数
lemmatizer = WordNetLemmatizer()

def clean_text(text, remove_stopwords=True):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# 加载数据
print("正在加载数据...")
train_df = pd.read_csv('train.csv')
valid_df = pd.read_csv('valid.csv')
test_df = pd.read_csv('test.csv')

# 清洗数据
print("正在清洗训练集数据...")
train_df['text'] = train_df['text'].apply(clean_text)
train_df['text'] = train_df['text'].apply(lemmatize_text)

print("正在清洗验证集数据...")
valid_df['text'] = valid_df['text'].apply(clean_text)
valid_df['text'] = valid_df['text'].apply(lemmatize_text)

print("正在清洗测试集数据...")
test_df['text'] = test_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(lemmatize_text)

# 检查空值
print("\n检查清洗后的数据是否有空值:")
print(f"训练集空值数量: {train_df['text'].isnull().sum()}")
print(f"验证集空值数量: {valid_df['text'].isnull().sum()}")
print(f"测试集空值数量: {test_df['text'].isnull().sum()}")

# 保存清洗后的数据
print("\n正在保存清洗后的数据...")
train_df.to_csv('train_cleaned.csv', index=False)
valid_df.to_csv('valid_cleaned.csv', index=False)
test_df.to_csv('test_cleaned.csv', index=False)

# 分析标签分布
print("\n正在分析标签分布...")
train_df = pd.read_csv('train_cleaned.csv')
valid_df = pd.read_csv('valid_cleaned.csv')
test_df = pd.read_csv('test_cleaned.csv')

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

print("\n数据清洗和标签分布分析已完成!")