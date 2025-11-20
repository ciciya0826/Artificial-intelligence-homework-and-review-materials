import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import spacy
from nltk.corpus import stopwords

# 1. 加载清洗后的数据（需提前完成数据清洗）
data_dir = "./"  # 替换为实际路径
train_df = pd.read_csv(f"{data_dir}/train_cleaned.csv")
valid_df = pd.read_csv(f"{data_dir}/valid_cleaned.csv")
test_df = pd.read_csv(f"{data_dir}/test_cleaned.csv")

# 2. 数据集大小统计
print("===== 数据集大小 =====")
print(f"训练集大小: {len(train_df)}")
print(f"开发集大小: {len(valid_df)}")
print(f"测试集大小: {len(test_df)}")

# 3. 训练集正负向情感句子数量
print("\n===== 训练集情感分布 =====")
positive_count = train_df["label"].value_counts().get(1, 0)
negative_count = train_df["label"].value_counts().get(0, 0)
print(f"正向情感句子数: {positive_count}")
print(f"负向情感句子数: {negative_count}")

# 4. 正负向情感中频率前十的词
def get_top_words(df, label, top_n=10):
    texts = df[df["label"] == label]["text"]
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(texts)
    word_counts = Counter(dict(zip(vectorizer.get_feature_names_out(), counts.sum(axis=0).tolist()[0])))
    return word_counts.most_common(top_n)

print("\n===== 频率前十的词 =====")
print("正向情感高频词:", get_top_words(train_df, 1))
print("负向情感高频词:", get_top_words(train_df, 0))

# 5. 计算PMI（点互信息）
def calculate_pmi(word_counts, total_words, class_counts, total_docs):
    pmi = {}
    for word, count in word_counts.items():
        p_word = count / total_words  # 词在类别中的频率
        p_class = class_counts / total_docs  # 类别在数据集中的频率
        p_word_class = count / total_docs  # 词和类别同时出现的频率
        if p_word * p_class == 0:
            pmi[word] = 0.0
        else:
            pmi[word] = np.log2(p_word_class / (p_word * p_class))  # PMI公式
    return pmi

# 统计正向情感PMI
positive_texts = train_df[train_df["label"] == 1]["text"]
positive_words = " ".join(positive_texts).split()
positive_word_counts = Counter(positive_words)
total_positive_words = len(positive_words)
total_docs = len(train_df)
positive_pmi = calculate_pmi(positive_word_counts, total_positive_words, positive_count, total_docs)
print("\n===== PMI前十大的词 =====")
print("正向情感PMI前十词:", sorted(positive_pmi.items(), key=lambda x: x[1], reverse=True)[:10])

# 统计负向情感PMI（同理）
negative_texts = train_df[train_df["label"] == 0]["text"]
negative_words = " ".join(negative_texts).split()
negative_word_counts = Counter(negative_words)
total_negative_words = len(negative_words)
negative_pmi = calculate_pmi(negative_word_counts, total_negative_words, negative_count, total_docs)
print("负向情感PMI前十词:", sorted(negative_pmi.items(), key=lambda x: x[1], reverse=True)[:10])

# 6. 词性分析（使用spaCy）
print("\n===== 词性分布 =====")
nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner"])  # 仅加载词性标注器
pos_counts = Counter()
for text in train_df["text"]:
    doc = nlp(text)
    for token in doc:
        pos_counts[token.pos_] += 1
print("前5大词性:", pos_counts.most_common(5))

# 7. 情感词分析（基于自定义词典）
stop_words = set(stopwords.words("english"))
positive_words = {"good", "great", "amazing", "excellent"}  # 自定义正向词典
negative_words = {"bad", "terrible", "awful", "horrible"}    # 自定义负向词典

def analyze_sentiment_words(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    return pos_count, neg_count

train_df["pos_words"], train_df["neg_words"] = zip(*train_df["text"].apply(analyze_sentiment_words))
print("\n平均正向情感词数:", train_df["pos_words"].mean())
print("平均负向情感词数:", train_df["neg_words"].mean())