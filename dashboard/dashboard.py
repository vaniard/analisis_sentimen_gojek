import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
import re
import requests
import csv
from io import StringIO
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- SETUP DASHBOARD ---
st.set_page_config(page_title="Dashboard Sentimen Ulasan Gojek", layout="wide")
st.title("📊 Dashboard Analisis Sentimen Ulasan Gojek")

# --- 1. LOAD DATA & LEXICON (Dengan Cache agar Cepat) ---
@st.cache_data
def load_lexicons():
    # Mengambil Leksikon Positif & Negatif InSet dari Github
    lexicon_pos = dict()
    lexicon_neg = dict()
    
    url_pos = 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv'
    resp_pos = requests.get(url_pos)
    if resp_pos.status_code == 200:
        reader = csv.reader(StringIO(resp_pos.text), delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                try: lexicon_pos[row[0]] = int(row[1])
                except ValueError: continue
                
    url_neg = 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv'
    resp_neg = requests.get(url_neg)
    if resp_neg.status_code == 200:
        reader = csv.reader(StringIO(resp_neg.text), delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                try: lexicon_neg[row[0]] = int(row[1])
                except ValueError: continue
                
    return lexicon_pos, lexicon_neg

@st.cache_data
def load_data():
    df = pd.read_csv("clean_data_ulasan_gojek.csv")
    df = df.dropna(subset=['text_final'])
    return df

@st.cache_data
def calculate_polarity(df, lex_pos, lex_neg):
    def get_sentiment(text):
        score = 0
        if isinstance(text, str):
            for word in text.split():
                if word in lex_pos: score += lex_pos[word]
                elif word in lex_neg: score += lex_neg[word]
        
        if score > 0: return score, 'positive'
        elif score < 0: return score, 'negative'
        else: return score, 'neutral'
        
    results = df['text_final'].apply(get_sentiment)
    df['polarity_score'] = [r[0] for r in results]
    df['polarity'] = [r[1] for r in results]
    return df

with st.spinner("Memuat Data dan Leksikon..."):
    lexicon_pos, lexicon_neg = load_lexicons()
    df_raw = load_data()
    df_clean = calculate_polarity(df_raw, lexicon_pos, lexicon_neg)


# --- 2. TRAINING MODEL UNTUK PREDIKSI INPUT ---
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df['text_final'])
    y = df['polarity']
    
    # Feature Selection 
    selector = SelectKBest(chi2, k=1000)
    X_selected = selector.fit_transform(X, y)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_selected, y)
    
    return vectorizer, selector, rf_model

with st.spinner("Menyiapkan Model Machine Learning..."):
    vectorizer, selector, model = train_model(df_clean)

# --- 3. NLP PREPROCESSING UNTUK TEKS BARU ---
@st.cache_resource
def get_nlp_tools():
    stemmer = StemmerFactory().create_stemmer()
    stopword = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, stopword

stemmer, stopword_remover = get_nlp_tools()

def preprocess_text(text):
    text = text.lower() # Casefolding
    text = re.sub(r'[^a-z\s]', '', text) # Cleaning
    text = stopword_remover.remove(text) # Filtering
    text = stemmer.stem(text) # Stemming
    return text

# --- 4. TABS UNTUK DASHBOARD ---
tab1, tab2, tab3 = st.tabs(["Distribusi Sentimen & Visualisasi", "Word Cloud & Top Words", "Uji Prediksi Sentimen Baru"])

with tab1:
    st.header("Visualisasi Distribusi Sentimen")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pie Chart Polarity")
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        sizes = df_clean['polarity'].value_counts()
        labels = sizes.index
        explode = [0.1] + [0] * (len(sizes) - 1)
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', explode=explode, textprops={'fontsize': 14}, shadow=True, colors=['#ff9999','#66b3ff','#99ff99'])
        ax1.set_title("Polarity Sentimen berdasarkan Data Ulasan")
        st.pyplot(fig1)
        
    with col2:
        st.subheader("Bar Chart Distribusi Data Ulasan")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.countplot(x='polarity', data=df_clean, palette='viridis', ax=ax2)
        for p in ax2.patches:
            ax2.annotate(format(p.get_height(), '.0f'), 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', xytext=(0, 9), textcoords='offset points')
        st.pyplot(fig2)
        
    st.subheader("Distribusi Panjang Teks (Text Length)")
    df_clean['text_length'] = df_clean['text_final'].apply(lambda x: len(str(x).split()))
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.histplot(df_clean['text_length'], bins=30, kde=True, color='teal', ax=ax3)
    ax3.set_title("Distribusi Text Length")
    st.pyplot(fig3)

with tab2:
    st.header("Top Words & Word Cloud")
    
    st.subheader("Top 20 Words berdasarkan TF-IDF")
    @st.cache_data
    def get_tfidf_top(df):
        vec = TfidfVectorizer()
        x = vec.fit_transform(df['text_final'].astype(str))
        df_tfidf = pd.DataFrame(x.toarray(), columns=vec.get_feature_names_out())
        df_sum = df_tfidf.sum().reset_index(name='result')
        return df_sum.sort_values('result', ascending=False).head(20)
    
    top_words = get_tfidf_top(df_clean)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='result', y='index', data=top_words, palette='magma', ax=ax4)
    st.pyplot(fig4)
    
    st.subheader("Word Cloud berdasarkan Kelas")
    col_wc1, col_wc2, col_wc3 = st.columns(3)
    
    def plot_wordcloud(text, title):
        wc = WordCloud(width=400, height=400, background_color='white', stopwords=STOPWORDS).generate(text)
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, pad=20)
        return fig
    
    pos_text = ' '.join(df_clean[df_clean['polarity'] == 'positive']['text_final'].astype(str))
    neg_text = ' '.join(df_clean[df_clean['polarity'] == 'negative']['text_final'].astype(str))
    neu_text = ' '.join(df_clean[df_clean['polarity'] == 'neutral']['text_final'].astype(str))
    
    if pos_text.strip():
        with col_wc1: st.pyplot(plot_wordcloud(pos_text, "Positive Words"))
    if neg_text.strip():
        with col_wc2: st.pyplot(plot_wordcloud(neg_text, "Negative Words"))
    if neu_text.strip():
        with col_wc3: st.pyplot(plot_wordcloud(neu_text, "Neutral Words"))

with tab3:
    st.header("Uji Prediksi Sentimen")
    st.markdown("Fitur ini akan memproses teks baru dengan TF-IDF dan memprediksi sentimen menggunakan *Random Forest* (sesuai *pipeline* dari notebook asli).")
    
    user_input = st.text_area("Masukkan ulasan Gojek atau kalimat baru:", placeholder="Ketik kalimat di sini...")
    
    if st.button("Prediksi Sentimen", type="primary"):
        if user_input.strip() == "":
            st.warning("⚠️ Silakan masukkan teks terlebih dahulu!")
        else:
            with st.spinner('Memproses teks (Cleansing, Stopword Removal, Stemming) dan memprediksi...'):
                # Preprocess Text
                cleaned_text = preprocess_text(user_input)
                st.info(f"**Teks setelah Pre-processing (text_final):** {cleaned_text}")
                
                # Transform to TF-IDF -> Feature Selection
                X_new = vectorizer.transform([cleaned_text])
                X_new_df = pd.DataFrame(X_new.toarray(), columns=vectorizer.get_feature_names_out())
                
                selector_new = selector.transform(np.array(X_new_df))
                
                # Predict
                prediction = model.predict(selector_new)[0]
                
                st.subheader("Hasil Prediksi:")
                if prediction == 'positive':
                    st.success("🟢 Sentimen kalimat adalah **POSITIVE**.")
                elif prediction == 'neutral':
                    st.info("🟡 Sentimen kalimat adalah **NEUTRAL**.")
                else:
                    st.error("🔴 Sentimen kalimat adalah **NEGATIVE**.")