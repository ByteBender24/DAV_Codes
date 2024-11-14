import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import os
import time
from scipy.stats import ttest_ind, f_oneway, shapiro, mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
import statsmodels.api as sm

# Streamlit Setup
st.title('Hate Speech Tweet Analysis Dashboard')

# Data Loading Function
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_train_data_v0.csv')
    return df

# Preprocessing Functions
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

@st.cache_data
def preprocess_text(df):
    # Check if preprocessed file already exists to avoid recomputing
    processed_file = 'final_cleaned_data.csv'
    if os.path.exists(processed_file):
        df = pd.read_csv(processed_file)
    else:
        # Data Preprocessing (if not already preprocessed)
        df['cleaned_tweet'] = df['cleaned_tweet'].astype(str)
        df['cleaned_tweet'] = df['cleaned_tweet'].str.replace(r'\b(url|user)\b', '', regex=True)
        df['cleaned_tweet'] = df['cleaned_tweet'].str.replace(r'[^\w\s]', '', regex=True)
        df['cleaned_tweet'] = df['cleaned_tweet'].str.lower()

        # Apply stopword removal, stemming, and lemmatization
        df['stopped'] = df['cleaned_tweet'].apply(remove_stopwords)
        df['stemmed_tweet'] = df['cleaned_tweet'].apply(stem_sentence)
        df['lemmatized_tweet'] = df['cleaned_tweet'].apply(lemmatize_sentence)

        # Save the processed data
        df.to_csv(processed_file, index=False)
    
    return df

@st.cache_data
def remove_stopwords(sentence):
    words = sentence.split()
    filtered = ' '.join(word for word in words if word not in stop_words)
    return filtered

@st.cache_data
def stem_sentence(sentence):
    words = sentence.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

@st.cache_data
def lemmatize_sentence(sentence):
    words = sentence.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# TF-IDF Preprocessing and Saving
@st.cache_data
def generate_tfidf(df):
    # Check if TF-IDF is already saved to avoid recomputation
    
    tfidf_file = 'tfidf.csv'
    if os.path.exists(tfidf_file):
        df_tfidf = pd.read_csv(tfidf_file)
    else:
        # Apply TF-IDF vectorization
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['lemmatized_tweet'])
        df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        # Save TF-IDF DataFrame
        df_tfidf.to_csv(tfidf_file, index=False)
    
    return df_tfidf

# Streamlit Sidebar - Select the analysis components
st.sidebar.header("Analysis Settings")
analysis_type = st.sidebar.selectbox("Select Analysis Type", 
                                    ["Data Overview", 
                                     "Tweet Length Distribution", 
                                     "Sentiment Analysis", 
                                     "TF-IDF Analysis", 
                                     "Topic Modeling", 
                                     "Word Cloud Analysis", 
                                     "Statistical Tests"])

# Load and preprocess data once
df = load_data()
df = preprocess_text(df)

# Data Overview
if analysis_type == "Data Overview":
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write(df.info())
    st.write(df.describe())

# Tweet Length Distribution
elif analysis_type == "Tweet Length Distribution":
    st.subheader("Tweet Length Distribution")
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(df['tweet_length'], bins=40, kde=True, color='skyblue')
    plt.title('Distribution of Tweet Lengths')
    plt.xlabel('Tweet Length (in words)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Sentiment Analysis
elif analysis_type == "Sentiment Analysis":
    st.subheader("Sentiment Score Distribution")
    plt.figure(figsize=(8, 6))
    plt.hist(df['sentiment_score'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Sentiment Score Distribution')
    plt.xlabel('Sentiment Score (Compound)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

# TF-IDF Analysis
elif analysis_type == "TF-IDF Analysis":
    st.subheader("Top 10 TF-IDF Words in Non-Hate vs Hate Tweets")

    # Generate TF-IDF values (only once, from saved file)
    df_tfidf = generate_tfidf(df)

    # Compute average TF-IDF score per word
    avg_positive_tfidf = df_tfidf.iloc[:len(df[df['subtask_a'] == 0])].mean(axis=0).sort_values(ascending=False).head(10)
    avg_negative_tfidf = df_tfidf.iloc[len(df[df['subtask_a'] == 0]):].mean(axis=0).sort_values(ascending=False).head(10)

    # Show bar plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    avg_positive_tfidf.plot(kind='bar', ax=axes[0], color='green')
    axes[0].set_title("Top 10 TF-IDF Words in No Hate")
    axes[0].set_xlabel("Words")
    axes[0].set_ylabel("Average TF-IDF Score")

    avg_negative_tfidf.plot(kind='bar', ax=axes[1], color='red')
    axes[1].set_title("Top 10 TF-IDF Words in Hate speech")
    axes[1].set_xlabel("Words")
    axes[1].set_ylabel("Average TF-IDF Score")

    st.pyplot(fig)

# Topic Modeling
elif analysis_type == "Topic Modeling":
    st.subheader("Topic Modeling Using LDA")

    # Generate LDA model
    # Replace NaN or non-string values with an empty string
    df['lemmatized_tweet'] = df['lemmatized_tweet'].apply(lambda x: str(x) if isinstance(x, str) else '')

    vectorizer = CountVectorizer(stop_words='english', min_df=5)
    dtm = vectorizer.fit_transform(df['lemmatized_tweet'])
    
    # Perform LDA
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(dtm)
    
    words = vectorizer.get_feature_names_out()
    for i, topic in enumerate(lda.components_):
        st.write(f"Topic {i + 1}:")
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [words[j] for j in top_words_idx]
        st.write(" ".join(top_words))

    st.subheader("Word Cloud of Topic 1")
    topic_1_words = ' '.join([words[j] for j in lda.components_[0].argsort()[-10:][::-1]])
    wordcloud = WordCloud(width=800, height=400).generate(topic_1_words)
    plt.figure(figsize=(20, 16))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Word Cloud Analysis
elif analysis_type == "Word Cloud Analysis":
    st.subheader("Word Cloud for Offensive and Non-Offensive Tweets")
    # Convert all non-string values in 'lemmatized_tweet' to empty string
    df['lemmatized_tweet'] = df['lemmatized_tweet'].apply(lambda x: str(x) if isinstance(x, str) else '')

    non_offensive_tweets = ' '.join(df[df['subtask_a'] == 0]['lemmatized_tweet'])
    offensive_tweets = ' '.join(df[df['subtask_a'] == 1]['lemmatized_tweet'])

    st.subheader("Non-Offensive Tweets Word Cloud")
    wordcloud_non_offensive = WordCloud(width=800, height=400).generate(non_offensive_tweets)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud_non_offensive, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    st.subheader("Offensive Tweets Word Cloud")
    wordcloud_offensive = WordCloud(width=800, height=400).generate(offensive_tweets)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud_offensive, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Statistical Tests
elif analysis_type == "Statistical Tests":
    st.subheader("Statistical Test Results")
    # You can implement t-tests here as in your original code

    # Load the necessary data and preprocessed DataFrame
    df_tfidf = generate_tfidf(df)  # Preprocess and get the TF-IDF DataFrame
    # Replace NaN or non-string values with an empty string
    df['lemmatized_tweet'] = df['lemmatized_tweet'].apply(lambda x: str(x) if isinstance(x, str) else '')
    
    non_offensive_reviews = df[df['subtask_a'] == 0]['lemmatized_tweet']
    offensive_reviews = df[df['subtask_a'] == 1]['lemmatized_tweet']
    
    # Create TF-IDF vectorizer and transform non-offensive and offensive reviews
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    non_offensive_tfidf = tfidf_vectorizer.fit_transform(non_offensive_reviews)
    offensive_tfidf = tfidf_vectorizer.fit_transform(offensive_reviews)
    
    # T-Test: Compare average TF-IDF scores between non-offensive and offensive tweets
    non_offensive_avg_tfidf = non_offensive_tfidf.mean(axis=0).A1
    offensive_avg_tfidf = offensive_tfidf.mean(axis=0).A1

    stat, p_value_ttest = ttest_ind(non_offensive_avg_tfidf, offensive_avg_tfidf)
    st.write(f"T-Test between Non-Offensive and Offensive Tweets:")
    st.write(f"T-statistic: {stat}")
    st.write(f"P-value: {p_value_ttest}")
    if p_value_ttest < 0.05:
        st.write("Conclusion: There is a statistically significant difference in the average TF-IDF scores between non-offensive and offensive tweets.")
    else:
        st.write("Conclusion: There is no statistically significant difference in the average TF-IDF scores between non-offensive and offensive tweets.")

    # ANOVA: Check if there is a significant difference in the TF-IDF scores
    f_statistic, p_value_anova = f_oneway(non_offensive_avg_tfidf, offensive_avg_tfidf)
    st.write(f"ANOVA Test for Average TF-IDF Scores:")
    st.write(f"F-statistic: {f_statistic}")
    st.write(f"P-value: {p_value_anova}")
    if p_value_anova < 0.05:
        st.write("Conclusion: There is a statistically significant difference in the average TF-IDF scores between non-offensive and offensive tweets.")
    else:
        st.write("Conclusion: There is no statistically significant difference in the average TF-IDF scores between non-offensive and offensive tweets.")
    
    # Shapiro-Wilk Test: Test for normality in 'sentiment_score' and 'tweet_length'
    stat_sentiment, p_value_sentiment = shapiro(df['sentiment_score'].dropna())
    stat_length, p_value_length = shapiro(df['tweet_length'].dropna())

    st.write(f"Shapiro-Wilk Test for Normality of Sentiment Score:")
    st.write(f"Statistic: {stat_sentiment}, P-value: {p_value_sentiment}")
    if p_value_sentiment < 0.05:
        st.write("Conclusion: The sentiment score is not normally distributed.")
    else:
        st.write("Conclusion: The sentiment score is normally distributed.")

    st.write(f"Shapiro-Wilk Test for Normality of Tweet Length:")
    st.write(f"Statistic: {stat_length}, P-value: {p_value_length}")
    if p_value_length < 0.05:
        st.write("Conclusion: The tweet length is not normally distributed.")
    else:
        st.write("Conclusion: The tweet length is normally distributed.")

    # Mann-Whitney U Test: Compare sentiment scores between non-offensive and offensive tweets
    non_offensive_scores = df[df['subtask_a'] == 0]['sentiment_score']
    offensive_scores = df[df['subtask_a'] == 1]['sentiment_score']

    stat_mannwhitney, p_value_mannwhitney = mannwhitneyu(non_offensive_scores, offensive_scores)
    st.write(f"Mann-Whitney U Test for Sentiment Scores between Non-Offensive and Offensive Tweets:")
    st.write(f"Statistic: {stat_mannwhitney}, P-value: {p_value_mannwhitney}")
    if p_value_mannwhitney < 0.05:
        st.write("Conclusion: There is a statistically significant difference in sentiment scores between non-offensive and offensive tweets.")
    else:
        st.write("Conclusion: There is no statistically significant difference in sentiment scores between non-offensive and offensive tweets.")

    # Visualizations for Distribution of TF-IDF scores and Sentiment Scores
    st.subheader("Visualizations for Statistical Results")

    # Visualize the TF-IDF distributions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(non_offensive_avg_tfidf, kde=True, color='blue')
    plt.title('TF-IDF Distribution for Non-Offensive Tweets')
    plt.xlabel('TF-IDF Score')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(offensive_avg_tfidf, kde=True, color='red')
    plt.title('TF-IDF Distribution for Offensive Tweets')
    plt.xlabel('TF-IDF Score')
    plt.ylabel('Frequency')

    plt.tight_layout()
    st.pyplot(plt)

    # Plot QQ plot for sentiment score distribution
    plt.figure(figsize=(8, 6))
    sm.qqplot(df['sentiment_score'].dropna(), line='45')
    plt.title('QQ Plot for Sentiment Score')
    st.pyplot(plt)

    # Plot QQ plot for tweet length distribution
    plt.figure(figsize=(8, 6))
    sm.qqplot(df['tweet_length'].dropna(), line='45')
    plt.title('QQ Plot for Tweet Length')
    st.pyplot(plt)