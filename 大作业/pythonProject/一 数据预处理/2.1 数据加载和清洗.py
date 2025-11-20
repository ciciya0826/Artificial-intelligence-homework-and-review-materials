import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer

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
train_df = pd.read_csv('train.csv')
valid_df = pd.read_csv('valid.csv')
test_df = pd.read_csv('test.csv')

# 清洗数据
train_df['text'] = train_df['text'].apply(clean_text)
train_df['text'] = train_df['text'].apply(lemmatize_text)

valid_df['text'] = valid_df['text'].apply(clean_text)
valid_df['text'] = valid_df['text'].apply(lemmatize_text)

test_df['text'] = test_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(lemmatize_text)

# 检查空值
print(f"训练集空值数量: {train_df['text'].isnull().sum()}")
print(f"验证集空值数量: {valid_df['text'].isnull().sum()}")
print(f"测试集空值数量: {test_df['text'].isnull().sum()}")

# 保存清洗后的数据
train_df.to_csv('train_cleaned.csv', index=False)
valid_df.to_csv('valid_cleaned.csv', index=False)
test_df.to_csv('test_cleaned.csv', index=False)