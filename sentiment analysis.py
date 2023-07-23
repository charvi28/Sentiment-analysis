#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


# Specify the file path
file_path = 'C:\\Users\\charv\\OneDrive\\Desktop\\Reviews.csv'


# In[4]:


data = pd.read_csv('C:\\Users\\charv\\OneDrive\\Desktop\\Reviews.csv')


# In[5]:


data.head()


# In[6]:


# Imports
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
# Product Scores
fig = px.histogram(data, x="Score")
fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Product Score')
fig.show()


# In[7]:


import nltk
from nltk.corpus import stopwords


# In[8]:


nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))


# In[9]:


get_ipython().system('pip install wordcloud')


# In[22]:


from wordcloud import WordCloud


# In[24]:


# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["br", "href"])
textt = " ".join(review for review in data.Text)
wordcloud = WordCloud(stopwords=stopwords).generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud11.png')
plt.show()


# In[26]:


#assign reviews with score > 3 as positive sentiment
# score < 3 negative sentiment
# remove score = 3
data = data[data['Score'] != 3]
data['sentiment'] = data['Score'].apply(lambda rating : +1 if rating > 3 else -1)


# In[27]:


data.head()


# In[28]:


# split df - positive and negative sentiment:
positive = data[data['sentiment'] == 1]
negative = data[data['sentiment'] == -1]


# In[29]:


stopwords = set(STOPWORDS)
stopwords.update(["br", "href","good","great"]) 
## good and great removed because they were included in negative sentiment
pos = " ".join(review for review in positive.Summary)
wordcloud2 = WordCloud(stopwords=stopwords).generate(pos)
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[33]:


neg = " ".join(str(review) for review in negative.Summary if not isinstance(review, float))
wordcloud3 = WordCloud(stopwords=stopwords).generate(neg)
plt.imshow(wordcloud3, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud33.png')
plt.show()


# In[34]:


data['sentimentt'] = data['sentiment'].replace({-1 : 'negative'})
data['sentimentt'] = data['sentimentt'].replace({1 : 'positive'})
fig = px.histogram(data, x="sentimentt")
fig.update_traces(marker_color="indianred",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Product Sentiment')
fig.show()


# In[35]:


data.head()


# In[36]:


#data cleaning
def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"'))
    return final
data['Text'] = data['Text'].apply(remove_punctuation)
data = data.dropna(subset=['Summary'])
data['Summary'] = data['Summary'].apply(remove_punctuation)


# In[37]:


#split data frame
dataNew = data[['Summary','sentiment']]
dataNew.head()


# In[39]:


import numpy as np


# In[40]:


# random split train and test data
index = data.index
data['random_number'] = np.random.randn(len(index))
train = data[data['random_number'] <= 0.8]
test = data[data['random_number'] > 0.8]


# In[41]:


# count vectorizer:
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['Summary'])
test_matrix = vectorizer.transform(test['Summary'])


# In[42]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[43]:


#split target and independent variables
X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']


# In[44]:


#fit model on data
lr.fit(X_train,y_train)


# In[45]:


#make a prediction
predictions = lr.predict(X_test)


# In[46]:


#testing
# find accuracy, precision, recall:
from sklearn.metrics import confusion_matrix,classification_report
new = np.asarray(y_test)
confusion_matrix(predictions,y_test)


# In[47]:


print(classification_report(predictions,y_test))


# In[4]:


git add .

