import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --- 1️⃣ API anahtarını kontrol et ---
if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("❌ GEMINI_API_KEY ortam değişkeni tanımlı değil. Terminalde: export GEMINI_API_KEY='ANAHTARIN'")

# --- 2️⃣ Dataset yükle ---
DATA_PATH = "turkish_law_dataset.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset bulunamadı: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# --- 3️⃣ Metin kolonunu belirle ---
# Eğer kolon isimleri farklıysa (örneğin 'text' yerine 'content' veya 'law_text'), burada değiştir:
TEXT_COLUMN = "text"
if TEXT_COLUMN not in df.columns:
    raise KeyError(f"'{TEXT_COLUMN}' kolonu bulunamadı! Kolon isimleri: {list(df.columns)}")

# --- 4️⃣ Metinleri LangChain Document formatına dönüştür ---
documents = []
for _, row in df.iterrows():
    content = str(row[TEXT_COLUMN])
    source = str(row.get("source", "Bilinmiyor"))
    documents.append(Document(page_content=content, metadata={"source": source}))

# --- 5️⃣ Chunking (parçalara ayırma) ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_docs = splitter.split_documents(documents)

print(f"✅ {len(split_docs)} adet metin parçası oluşturuldu.")

# --- 6️⃣ Embedding modeli ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",  # ⚠️ dikkat: model ismi başına 'models/' eklendi
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# --- 7️⃣ Chroma veritabanını oluştur ve kaydet ---
DB_DIR = "./chroma_db"
db = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=DB_DIR
)
db.persist()

print(f"✅ Chroma veritabanı oluşturuldu ve '{DB_DIR}' klasörüne kaydedildi.")
