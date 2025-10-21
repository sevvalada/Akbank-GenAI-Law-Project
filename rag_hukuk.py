# =========================================================================
# 1. KÜTÜPHANE İMPORTLARI (Gerekli Modüllerin Yüklenmesi)
# =========================================================================
import os 
import pandas as pd # Veri setini (CSV) okumak için.
from langchain.schema.document import Document # Veri parçalarını LangChain formatına dönüştürmek için.
from langchain.chains import RetrievalQA # RAG zincirini kurmak ve çalıştırmak için.
from langchain_community.vectorstores import Chroma # Vektör veritabanı olarak Chroma'yı kullanmak için.
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings # Gemini modelleri ve Embeddings için.

# =========================================================================
# 2. GEMINI MODELLERİNİ TANIMLAMA (API Erişimi ve Model Hazırlığı)
# =========================================================================
GEMINI_API_KEY_VALUE = os.getenv('GEMINI_API_KEY')

# API Anahtarı Kontrolü. Anahtar yoksa programı durdurur.
if not GEMINI_API_KEY_VALUE:
    raise ValueError("HATA: GEMINI_API_KEY ortam değişkeni tanımlı değil. Lütfen terminalde 'export GEMINI_API_KEY=\"SİZİN_ANAHTARINIZ\"' (veya set) komutunu çalıştırmayı unutmayın.")

print("Gemini API Anahtarı ortam değişkeninden başarıyla çekildi.")

# Generation Model (Cevap Üretimi): RAG zincirinde nihai cevabı üretecek olan model.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2, # Düşük sıcaklık, hukuki cevaplar için kesinliği artırır.
    google_api_key=GEMINI_API_KEY_VALUE    
)

# Embedding Model (Vektörleştirme): Metinleri sayısal vektörlere dönüştürecek model.
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=GEMINI_API_KEY_VALUE,
    # Hata önleme çözümünü koruyoruz: API'ye aynı anda gönderilen belge sayısını sınırlayarak (Batch Size 50) olası API limit hatalarını engeller.
    batch_size=50 
)
print("Gemini Modelleri ve Embedding başarıyla tanımlandı (Batch Size 50).")

# =========================================================================
# 3. VERİ YÜKLEME VE LANGCHAIN DOCUMENT FORMATINA DÖNÜŞTÜRME (TAM VERİ SETİ)
# =========================================================================

FILE_NAME = "turkish_law_dataset.csv" 
print(f"'{FILE_NAME}' yükleniyor...")

try:
    # Veri setini yükle
    df = pd.read_csv(FILE_NAME)
    SUTUN_CONTEXT = 'context'
    SUTUN_KAYNAK = 'kaynak'
    SUTUN_TUR = 'veri türü'
except FileNotFoundError:
    # Dosya bulunamazsa hata mesajı verip programı sonlandır.
    print(f"\nHATA: '{FILE_NAME}' dosyası bulunamadı. Lütfen dosyanın '{os.getcwd()}' klasöründe olduğundan emin olun.")
    exit()

documents = []
# KRİTİK DÜZELTME: Sınırlama kaldırıldı, TÜM VERİ İŞLENİYOR!
# DataFrame'deki her bir satırı LangChain'in anlayacağı Document nesnesine dönüştürür.
for index, row in df.iterrows():
    doc = Document(
        page_content=row[SUTUN_CONTEXT], # Metnin ana içeriği (Hukuk metinleri).
        metadata={ # Metnin ek bilgileri (Filtreleme ve kaynak gösterme için kritik).
            "source": row[SUTUN_KAYNAK], # Hangi kanuna ait olduğu (Anayasa, TMK, TCK, vb.).
            "veri_turu": row[SUTUN_TUR],
            "soru": row['soru'], # Test/Benchmark için tutulan orijinal soru.
            "cevap": row['cevap'] # Test/Benchmark için tutulan orijinal cevap.
        }
    )
    documents.append(doc)

print(f"Toplam yüklü ve işlenmeye hazır belge (context) sayısı: {len(documents)}. (Tüm veri seti kullanılıyor.)")

# =========================================================================
# 4. EMBEDDING VE VEKTÖR VERİTABANI OLUŞTURMA (RAG ÇEKİRDEĞİ)
# =========================================================================

print("\nEmbedding (Vektörleştirme) işlemi başlatılıyor... (Bu adım, API limit sorununu çözmek için Batch 50 ile çalışıyor.)")
# Chroma.from_documents: Tüm metinleri vektörlere dönüştürür, vektör veritabanına kaydeder ve diske kaydeder (persists).
vectorstore = Chroma.from_documents(
    documents=documents, # Vektörleştirilecek LangChain Document listesi.
    embedding=embeddings, # Vektörleştirmeyi yapacak Gemini modeli.
    persist_directory="./chroma_db" # Vektörlerin kalıcı olarak kaydedileceği klasör.
)
print("Embedding ve Vektör Veritabanı kaydı tamamlandı.")

# Retriever'ı Tanımlama: Vektör veritabanından sorguya en yakın dokümanları çekecek mekanizma.
retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # En yakın 4 belgeyi çekmek üzere ayarlanır.
print("Retriever başarıyla tanımlandı.")

# =========================================================================
# 5. RAG ZİNCİRİNİ KURMA VE TEST (RAG İş Akışını Çalıştırma)
# =========================================================================

print("\nRAG Zinciri (RetrievalQA) kuruluyor...")
# RetrievalQA: LangChain'de retrieval (alma) ve generation (üretme) adımlarını birleştiren standart zincir.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, # Cevap üretimi için kullanılacak LLM.
    chain_type="stuff", # Çekilen dokümanları tek bir prompt'a koyma yöntemi.
    retriever=retriever, # Dokümanları çekecek retriever.
    return_source_documents=True # Cevapla birlikte kullanılan kaynakları da döndürmeyi sağlar.
)
print("RAG Zinciri başarıyla kuruldu.")

# --- ÖRNEK TEST ---
sorgu = "Anayasa, Türkiye Cumhuriyeti'nin hangi milliyetçilik anlayışını temel alıyor?" 

print(f"\n--- TEST BAŞLATILIYOR ---\nSorgu: {sorgu}")
# Zinciri çalıştırma
result = qa_chain.invoke({"query": sorgu}) 

# Cevabı ve Alıntıları Gösterme
print("\n--- CHATBOT CEVABI ---")
print(result['result'])

print("\n--- KULLANILAN KAYNAKLAR (Alıntılar) ---")
# Çekilen kaynak dokümanları gösterilir (kaynak adı ve metin içeriğinin bir kısmı).
for i, doc in enumerate(result['source_documents']):
    kaynak_adi = doc.metadata.get('source', 'Bilinmiyor')
    print(f"{i+1}. Kaynak: {kaynak_adi}")
    print(f"   Metin (Bağlam): {doc.page_content[:150]}...") # Metnin ilk 150 karakteri gösterilir.
    print("-" * 20)

# KOD SONU