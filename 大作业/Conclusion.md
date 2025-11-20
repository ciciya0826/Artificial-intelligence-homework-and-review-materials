# 人工智能课程项目：IMDB情感分类

IMDB情感分类：标准的短文本英语文本二义分类问题

# 一：数据预处理

### **1. 数据加载**

首先加载数据集，检查数据的基本结构和内容。

```python
import pandas as pd

# 加载数据集
train_df = pd.read_csv('train.csv')
valid_df = pd.read_csv('valid.csv')
test_df = pd.read_csv('test.csv')

# 查看数据的前几行
print(train_df.head())
print(valid_df.head())
print(test_df.head())

# 检查数据的基本信息
print(train_df.info())
print(valid_df.info())
print(test_df.info())
```



### **2. 数据清洗**

#### **2.1 定义更全面的文本清洗函数**

改进后的清洗函数包括：

- 转换为小写
- 去除HTML标签
- 去除特殊字符和标点符号
- 去除多余的空格
- 去除数字
- 去除停用词
- 还原词根

```python
import re
from nltk.corpus import stopwords
import nltk

# 下载NLTK停用词（如果未下载）
nltk.download('stopwords')

# 定义改进后的文本清洗函数
def clean_text(text, remove_stopwords=True):
    # 转换为小写
    text = text.lower()
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 去除特殊字符和标点符号
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 去除数字（可选）
    text = re.sub(r'\d+', '', text)
    # 去除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    # 去除停用词（可选）
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
    return text
```

#### **2.2 应用清洗函数到数据集**

对训练集、验证集和测试集应用清洗函数。

```python
# 对训练集、验证集和测试集进行清洗
train_df['text'] = train_df['text'].apply(clean_text)
valid_df['text'] = valid_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(clean_text)

# 查看清洗后的数据
print("训练集清洗后的示例:")
print(train_df['text'].head())

print("验证集清洗后的示例:")
print(valid_df['text'].head())

print("测试集清洗后的示例:")
print(test_df['text'].head())
```

#### **2.3 检查清洗后的数据**

确保清洗后的数据没有丢失或损坏。

```python
# 检查训练集、验证集和测试集是否包含空值
print("训练集空值检查:")
print(train_df['text'].isnull().sum())

print("验证集空值检查:")
print(valid_df['text'].isnull().sum())

print("测试集空值检查:")
print(test_df['text'].isnull().sum())
```



#### **2.4 进一步优化**

- **词形还原（Lemmatization）**：将单词还原为词根形式。
- **词干提取（Stemming）**：将单词缩减为词干形式。
- **自定义停用词列表**：根据数据集特点添加或移除停用词。

这里我们小组选择词形还原：

```python
from nltk.stem import WordNetLemmatizer

# 下载WordNet（如果未下载）
nltk.download('wordnet')

# 初始化词形还原器
lemmatizer = WordNetLemmatizer()

# 定义词形还原函数
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# 对数据集应用词形还原
train_df['text'] = train_df['text'].apply(lemmatize_text)
valid_df['text'] = valid_df['text'].apply(lemmatize_text)
test_df['text'] = test_df['text'].apply(lemmatize_text)
```



#### **2.5  保存清洗后的数据**

将清洗后的数据保存到文件中，以便后续使用。

```python
# 保存清洗后的数据
train_df.to_csv('train_cleaned.csv', index=False)
valid_df.to_csv('valid_cleaned.csv', index=False)
test_df.to_csv('test_cleaned.csv', index=False)
```



### **3. 标签分布分析**

#### **3.1 加载清洗后的数据**

首先加载已经清洗并保存的数据。

```python
import pandas as pd

# 加载清洗后的数据
train_df = pd.read_csv('train_cleaned.csv')
valid_df = pd.read_csv('valid_cleaned.csv')
test_df = pd.read_csv('test_cleaned.csv')
```

#### **3.2 检查标签分布**

使用`value_counts()`方法查看训练集、验证集和测试集的标签分布。

```python
# 训练集标签分布
print("训练集标签分布:")
print(train_df['label'].value_counts())

# 验证集标签分布
print("验证集标签分布:")
print(valid_df['label'].value_counts())

# 测试集标签分布
print("测试集标签分布:")
print(test_df['label'].value_counts())
```

#### **3.3 可视化标签分布**

使用可视化工具（`seaborn`好看）更直观地展示标签分布。

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set(style="whitegrid")

# 定义绘制标签分布的函数
def plot_label_distribution(df, title):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df, palette='Set2')
    plt.title(title)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()

# 绘制训练集、验证集和测试集的标签分布
plot_label_distribution(train_df, '训练集标签分布')
plot_label_distribution(valid_df, '验证集标签分布')
plot_label_distribution(test_df, '测试集标签分布')
```



#### **3.4 分析结果**

根据标签分布的输出和可视化结果，分析数据集的平衡性：

这里数据集很规整，**标签分布均匀**（例如正负样本比例接近1:1），数据集是平衡的，可以直接用于训练。



#### **3.5 保存分析结果**

将标签分布的分析结果保存到文件中

```python
# 保存标签分布结果
label_distribution = {
    'train': train_df['label'].value_counts().to_dict(),
    'valid': valid_df['label'].value_counts().to_dict(),
    'test': test_df['label'].value_counts().to_dict()
}

import json
with open('label_distribution.json', 'w') as f:
    json.dump(label_distribution, f, indent=4)
```



## 4.特征提取：

Task1：

1）训练集、开发集和测试集分别有多大？

2）训练集中有多少正向情感和负向情感的句子？

3）训练集中每种情感中频率前十的词都有什么？PMI前十大的词都有什么？

4）训练集中的用词有什么特点？都是什么词性？他们表达任何情感信息吗？

### **1. 数据集大小**

```python
# 加载数据集
train_df = pd.read_csv('train_cleaned.csv')
valid_df = pd.read_csv('valid_cleaned.csv')
test_df = pd.read_csv('test_cleaned.csv')

# 数据集大小
print(f"训练集大小: {len(train_df)}")
print(f"开发集大小: {len(valid_df)}")
print(f"测试集大小: {len(test_df)}")
```

------

### **2. 训练集中正向和负向情感的句子数量**

```python
# 统计训练集中正向和负向情感的句子数量
positive_count = train_df['label'].value_counts().get(1, 0)
negative_count = train_df['label'].value_counts().get(0, 0)

print(f"训练集中正向情感的句子数量: {positive_count}")
print(f"训练集中负向情感的句子数量: {negative_count}")
```

------

### **3. 训练集中每种情感中频率前十的词**

```python
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# 获取正向和负向情感的文本
positive_texts = train_df[train_df['label'] == 1]['text']
negative_texts = train_df[train_df['label'] == 0]['text']

# 统计正向情感中频率前十的词
positive_vectorizer = CountVectorizer()
positive_counts = positive_vectorizer.fit_transform(positive_texts)
positive_word_counts = Counter(dict(zip(positive_vectorizer.get_feature_names_out(), positive_counts.sum(axis=0).tolist()[0])))
print("正向情感中频率前十的词:", positive_word_counts.most_common(10))

# 统计负向情感中频率前十的词
negative_vectorizer = CountVectorizer()
negative_counts = negative_vectorizer.fit_transform(negative_texts)
negative_word_counts = Counter(dict(zip(negative_vectorizer.get_feature_names_out(), negative_counts.sum(axis=0).tolist()[0])))
print("负向情感中频率前十的词:", negative_word_counts.most_common(10))
```

------

### **4. 训练集中每种情感中 PMI 前十大的词**

PMI（Pointwise Mutual Information）用于衡量词与情感类别之间的关联性。

```python
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# 计算 PMI
def calculate_pmi(word_counts, total_words, class_counts, total_docs):
    pmi = {}
    for word, count in word_counts.items():
        p_word = count / total_words
        p_class = class_counts / total_docs
        p_word_class = count / total_words
        pmi[word] = np.log2(p_word_class / (p_word * p_class))
    return pmi

# 统计正向情感中 PMI 前十大的词
total_positive_words = sum(positive_word_counts.values())
total_negative_words = sum(negative_word_counts.values())
total_docs = len(train_df)

positive_pmi = calculate_pmi(positive_word_counts, total_positive_words, positive_count, total_docs)
print("正向情感中 PMI 前十大的词:", sorted(positive_pmi.items(), key=lambda x: x[1], reverse=True)[:10])

# 统计负向情感中 PMI 前十大的词
negative_pmi = calculate_pmi(negative_word_counts, total_negative_words, negative_count, total_docs)
print("负向情感中 PMI 前十大的词:", sorted(negative_pmi.items(), key=lambda x: x[1], reverse=True)[:10])
```

------

### **5. 训练集中的用词特点**

#### **词性分析**

使用 `spaCy` 进行词性标注，统计训练集中词的词性分布。

```python
import spacy

# 加载 spaCy 模型
nlp = spacy.load("en_core_web_sm")

# 统计训练集中词的词性分布
pos_counts = Counter()
for text in train_df['text']:
    doc = nlp(text)
    for token in doc:
        pos_counts[token.pos_] += 1

print("训练集中词的词性分布:", pos_counts.most_common())
```

#### **情感分析**

使用情感词典分析训练集中词的情感倾向。

```python
# 统计训练集中词的情感倾向
sentiment_counts = Counter()
for text in train_df['text']:
    words = text.lower().split()
    for word in words:
        if word in positive_words:
            sentiment_counts["positive"] += 1
        elif word in negative_words:
            sentiment_counts["negative"] += 1

print("训练集中词的情感倾向:", sentiment_counts)
```



## 二：模型构建

Task2任务要求：

1. **对使用词频作为特征的逻辑回归模型进行特征选择**，例如选择最好的前 200 和前 2000 个特征，并训练两个不同的模型。
2. **在逻辑回归模型基础上，设计至少两个新的特征**，并训练一个新的模型。



### **1. 使用词频作为特征的逻辑回归模型**

#### **1.1 加载数据**

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 加载清洗后的数据
train_df = pd.read_csv('train_cleaned.csv')
test_df = pd.read_csv('test_cleaned.csv')

# 获取文本和标签
train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()

test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist()
```

#### **1.2 使用 TF-IDF 向量化文本**

```python
# 使用 TF-IDF 向量化文本
vectorizer = TfidfVectorizer(max_features=5000)  # 限制最大特征数为 5000
X_train_tfidf = vectorizer.fit_transform(train_texts)
X_test_tfidf = vectorizer.transform(test_texts)
```

#### **1.3 特征选择：选择最好的前 200 和前 2000 个特征**

```python
from sklearn.feature_selection import SelectKBest, chi2

# 选择最好的前 200 个特征
selector_200 = SelectKBest(chi2, k=200)
X_train_200 = selector_200.fit_transform(X_train_tfidf, train_labels)
X_test_200 = selector_200.transform(X_test_tfidf)

# 选择最好的前 2000 个特征
selector_2000 = SelectKBest(chi2, k=2000)
X_train_2000 = selector_2000.fit_transform(X_train_tfidf, train_labels)
X_test_2000 = selector_2000.transform(X_test_tfidf)
```

#### **1.4 训练逻辑回归模型**

```python
# 训练使用前 200 个特征的逻辑回归模型
lr_model_200 = LogisticRegression(max_iter=1000)
lr_model_200.fit(X_train_200, train_labels)

# 训练使用前 2000 个特征的逻辑回归模型
lr_model_2000 = LogisticRegression(max_iter=1000)
lr_model_2000.fit(X_train_2000, train_labels)

# 评估模型
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    print("分类报告:\n", classification_report(y_test, y_pred))

print("使用前 200 个特征的模型性能:")
evaluate_model(lr_model_200, X_test_200, test_labels)

print("使用前 2000 个特征的模型性能:")
evaluate_model(lr_model_2000, X_test_2000, test_labels)
```

------

### **2. 设计新的特征并训练模型**

#### **2.1 设计新特征**

以下是两个新的特征设计思路：

1. **文本长度**：文本的字符数。
2. **情感词数量**：文本中包含的情感词数量（使用情感词典）。

```python
# 定义情感词典（扩充）
# 正面情感词
positive_words = {
    "good", "great", "excellent", "happy", "love", "amazing", "wonderful", "fantastic", 
    "awesome", "brilliant", "perfect", "fabulous", "superb", "outstanding", "delightful", 
    "joyful", "pleased", "ecstatic", "thrilled", "glad", "blissful", "cheerful", 
    "content", "grateful", "optimistic", "positive", "satisfied", "elated", "euphoric", 
    "jubilant", "radiant", "serene", "triumphant", "upbeat", "victorious", "admirable", 
    "charming", "enjoyable", "favorable", "heartwarming", "inspiring", "magnificent", 
    "marvelous", "remarkable", "splendid", "stellar", "stupendous", "terrific", 
    "admiration", "affection", "bliss", "euphoria", "gratitude", "joy", "love", 
    "passion", "pleasure", "pride", "satisfaction", "triumph", "zeal"
}
# 反面情感词
negative_words = {
    "bad", "terrible", "awful", "sad", "hate", "horrible", "dreadful", "miserable", 
    "disappointing", "unhappy", "angry", "annoyed", "frustrated", "irritated", 
    "depressed", "gloomy", "heartbroken", "hopeless", "lonely", "melancholy", 
    "pessimistic", "sorrowful", "upset", "worried", "bitter", "despair", "disgust", 
    "envy", "fear", "grief", "guilt", "hatred", "jealousy", "regret", "shame", 
    "suffering", "tragic", "unfortunate", "agony", "anxiety", "desperation", 
    "discontent", "displeasure", "distress", "gloom", "heartache", "misery", 
    "pain", "resentment", "sadness", "torment", "unhappiness", "woe", "wrath"
}

# 计算新特征
def extract_new_features(texts):
    lengths = [len(text) for text in texts]  # 文本长度
    sentiment_counts = []
    for text in texts:
        words = text.split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        sentiment_counts.append(pos_count - neg_count)  # 情感词数量
    return lengths, sentiment_counts

# 提取训练集和测试集的新特征
train_lengths, train_sentiment_counts = extract_new_features(train_texts)
test_lengths, test_sentiment_counts = extract_new_features(test_texts)
```

#### **2.2 将新特征与 TF-IDF 特征结合**

```python
from scipy.sparse import hstack

# 将新特征与 TF-IDF 特征结合
X_train_new = hstack([X_train_tfidf, np.array(train_lengths).reshape(-1, 1), np.array(train_sentiment_counts).reshape(-1, 1)])
X_test_new = hstack([X_test_tfidf, np.array(test_lengths).reshape(-1, 1), np.array(test_sentiment_counts).reshape(-1, 1)])
```

#### **2.3 训练新的逻辑回归模型**

```python
# 训练新的逻辑回归模型
lr_model_new = LogisticRegression(max_iter=1000)
lr_model_new.fit(X_train_new, train_labels)

# 评估模型
print("使用新特征的模型性能:")
evaluate_model(lr_model_new, X_test_new, test_labels)
```



## **搭建朴素贝叶斯和逻辑回归模型**

接下来，我们小组使用 **朴素贝叶斯（Naive Bayes）** 和 **逻辑回归（Logistic Regression）** 模型进行情感分类。

#### **2.1 朴素贝叶斯模型**

朴素贝叶斯是一种基于贝叶斯定理的分类算法，假设特征之间相互独立。它适合处理高维稀疏数据，如文本数据。

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 使用 TF-IDF 向量化文本（如果未使用 DistilBERT 嵌入）
from sklearn.feature_extraction.text import TfidfVectorizer

# 假设使用 DistilBERT 嵌入
X_train = np.load('train_embeddings.npy')
y_train = train_df['label'].values

# 初始化朴素贝叶斯模型
nb_model = MultinomialNB()

# 训练模型
nb_model.fit(X_train, y_train)

# 预测
y_pred = nb_model.predict(X_train)

# 评估模型
print("朴素贝叶斯模型训练集准确率:", accuracy_score(y_train, y_pred))
print("分类报告:\n", classification_report(y_train, y_pred))
```

#### **2.2 逻辑回归模型**

逻辑回归是一种线性分类模型，适合处理二分类任务。它能够捕捉特征之间的相关性，且模型解释性强。

```python
from sklearn.linear_model import LogisticRegression

# 初始化逻辑回归模型
lr_model = LogisticRegression(max_iter=1000)

# 训练模型
lr_model.fit(X_train, y_train)

# 预测
y_pred = lr_model.predict(X_train)

# 评估模型
print("逻辑回归模型训练集准确率:", accuracy_score(y_train, y_pred))
print("分类报告:\n", classification_report(y_train, y_pred))
```

#### **2.3 使用验证集评估模型**

为了更准确地评估模型性能，建议使用验证集进行测试。

```python
# 生成验证集嵌入（假设已生成并保存为 valid_embeddings.npy）
X_valid = np.load('valid_embeddings.npy')
y_valid = valid_df['label'].values

# 使用朴素贝叶斯模型预测验证集
y_pred_nb = nb_model.predict(X_valid)
print("朴素贝叶斯模型验证集准确率:", accuracy_score(y_valid, y_pred_nb))

# 使用逻辑回归模型预测验证集
y_pred_lr = lr_model.predict(X_valid)
print("逻辑回归模型验证集准确率:", accuracy_score(y_valid, y_pred_lr))
```



# 调用大模型处理：

我们将之前处理好的数据（如 `train_cleaned.csv` 或 `train_embeddings.npy`）直接用于 GPT-4 的情感分类：

这里是这样的，我有密钥，所以用GPT，没有密钥建议换DeepSeek:

### **1. 加载清洗后的数据**

首先加载我们之前处理好的数据（如 `train_cleaned.csv`）。

```python
import pandas as pd

# 加载清洗后的数据
train_df = pd.read_csv('train_cleaned.csv')
texts = train_df['text'].tolist()  # 获取所有文本
labels = train_df['label'].tolist()  # 获取所有标签
```

------

### **2. 使用 GPT-4 进行批量情感分类**

将 `texts` 列表中的文本批量传递给 GPT-4 进行情感分类，并将结果保存到新的列中。

```python
import openai

# 定义 GPT-4 情感分类函数
def query_gpt4_for_sentiment(text):
    openai.api_key = "your-api-key"  # 替换为我们小组的 API 密钥
    openai.api_base = 'https://api.openai.com/v1'  # 使用官方 API 地址

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 使用 GPT-4 模型
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Classify the sentiment of the following text as 'positive' or 'negative': {text}"}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return str(e)

# 对每条文本进行情感分类
gpt4_labels = []
for text in texts:
    sentiment = query_gpt4_for_sentiment(text)
    gpt4_labels.append(1 if sentiment.lower() == 'positive' else 0)  # 将结果转换为 1（正面）或 0（负面）

# 将 GPT-4 的分类结果添加到 DataFrame 中
train_df['gpt4_label'] = gpt4_labels

# 保存结果到新的 CSV 文件
train_df.to_csv('train_with_gpt4_labels.csv', index=False)
```



1. **使用一种 Prompt 设计策略在一种大语言模型上测试至少 200 条测试集中的数据**。
2. **设计 Prompt 使大语言模型产生结构化输出，测试至少 20 条测试集中的数据**。

### **1. 测试至少 200 条测试集数据**

#### **Prompt 设计策略**

- **任务**：情感分类。
- **Prompt**：`"Classify the sentiment of the following text as 'positive' or 'negative': {text}"`

#### **代码实现**

```python
import openai
import time

# 定义 GPT-4 情感分类函数
def query_gpt4_for_sentiment(text):
    openai.api_key = "your-api-key"  # 替换为我们小组的 API 密钥
    openai.api_base = 'https://api.openai.com/v1'  # 使用官方 API 地址

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 使用 GPT-4 模型
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Classify the sentiment of the following text as 'positive' or 'negative': {text}"}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return str(e)

# 对测试集进行情感分类（至少 200 条）
batch_size = 10  # 每批处理 10 条文本
delay = 1  # 每次请求之间的延迟（秒）
test_gpt4_labels = []

for i in range(0, 200, batch_size):  # 仅测试前 200 条数据
    batch = test_texts[i:i + batch_size]
    for text in batch:
        sentiment = query_gpt4_for_sentiment(text)
        test_gpt4_labels.append(1 if sentiment.lower() == 'positive' else 0)
    time.sleep(delay)  # 添加延迟以避免速率限制

# 评估 GPT-4 的分类结果
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(test_labels[:200], test_gpt4_labels)
print(f"GPT-4 分类准确率: {accuracy:.4f}")
print("分类报告:\n", classification_report(test_labels[:200], test_gpt4_labels))
```

------

### **2. 测试至少 20 条测试集数据，生成结构化输出**

#### **Prompt 设计策略**

- **任务**：生成结构化输出，包含主题和情感。

- **Prompt**：

  ```
  Analyze the following text and provide structured output:
  1. Topic: What is the main topic of the text?
  2. Sentiment: Is the sentiment 'positive' or 'negative'?
  Text: {text}
  ```

#### **代码实现**

```python
# 定义 GPT-4 结构化输出函数
def query_gpt4_for_structured_output(text):
    openai.api_key = "your-api-key"  # 替换为我们小组的 API 密钥
    openai.api_base = 'https://api.openai.com/v1'  # 使用官方 API 地址

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 使用 GPT-4 模型
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Analyze the following text and provide structured output:\n1. Topic: What is the main topic of the text?\n2. Sentiment: Is the sentiment 'positive' or 'negative'?\nText: {text}"}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return str(e)

# 对测试集进行结构化输出（至少 20 条）
structured_outputs = []
for i in range(20):  # 仅测试前 20 条数据
    text = test_texts[i]
    output = query_gpt4_for_structured_output(text)
    structured_outputs.append(output)
    print(f"Text: {text}")
    print(f"Structured Output:\n{output}\n")
    time.sleep(delay)  # 添加延迟以避免速率限制
```



完整项目地址：需要GPU，或者修改代码（下周我再传一份CPU的版本）



ModelScope:[IMDB情感分类2025CCNU大二下人工智能大作业（项目完整版） · 数据集](https://www.modelscope.cn/datasets/David810/IMDB_Classificatiion_CCNU_2025_AI_Project/summary)

GitHub:[David-88/CCNU_2025_AI_IMDB_CLassification_Project: CCNU_2025_AI_IMDB_CLassification_Project_For_HomeWork](https://github.com/David-88/CCNU_2025_AI_IMDB_CLassification_Project)
