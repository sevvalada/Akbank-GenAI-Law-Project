import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Dict, Any
from langchain.retrievers.multi_query import MultiQueryRetriever # MultiQuery için gerekli

# =================================================================
# YARDIMCI FONKSİYON: SORGUYU KATEGORİZE ETME (Preprocessing/Routing Katmanı)
# =================================================================
def get_source_filter(llm_categorizer: ChatGoogleGenerativeAI, query: str) -> str:
    """
    Gemini modelini kullanarak kullanıcı sorgusunun ait olduğu hukuki kaynağı belirler.
    Bu, Retrieval alanını daraltır (Pre-filtering/Source Routing).
    """
    source_list = ["Türkiye Cumhuriyeti Anayasası", "Türk Medeni Kanunu", "Türk Ceza Kanunu", "Borçlar Kanunu", "İcra ve İflas Kanunu", "Diğer"]
    source_prompt = PromptTemplate(
        template=f"""Kullanıcı sorgusunu analiz et ve sorgunun en çok hangi hukuki kaynağa ait olduğunu belirle.
        Sadece aşağıdaki kaynaklardan birini cevap olarak döndür: {', '.join(source_list)}. 
        Eğer sorgu bu kaynaklardan birine ait değilse, 'Diğer' olarak cevap ver. 
        Sorgu: {query}
        Kaynak:""",
        input_variables=["query"]
    )
    
    # LLMChain: Modeli basit bir prompt ile çalıştırarak kategoriyi elde etme zinciri.
    chain = LLMChain(llm=llm_categorizer, prompt=source_prompt)
    
    try:
        # LLM'den gelen cevabı temizleyip, tanımlı kaynak listesiyle karşılaştırır.
        response = chain.run(query=query).strip()
        if response in source_list:
            return response
        else:
            return None # Kaynak, tanımlı listede yoksa veya kategori 'Diğer' ise None döner.
    except Exception:
        return None 

# =================================================================
# 1. ORTAM KONTROLÜ VE RAG PIPELINE YÜKLEMESİ (Kaynak Yönetimi)
# =================================================================

# Ortam değişkeni kontrolü: API anahtarının tanımlı olup olmadığını kontrol eder.
if 'GEMINI_API_KEY' not in os.environ:
    st.error("HATA: GEMINI_API_KEY ortam değişkeni tanımlı değil. Lütfen terminalde ayarlayın (export/set).")
    st.stop()

# RAG Prompt Şablonu (Gelişmiş Yorumlama) - LLM'in cevabı oluşturmak için kullanacağı talimat.
RAG_PROMPT_TEMPLATE = """
Sen, Türk hukuku metinleri hakkında cevaplar üreten bir asistansın.
Aşağıdaki 'Bağlam' kısmında sana sunulan hukuki metin parçalarını kullanarak, sadece bu metinlere dayanarak (ek bilgi eklemeden) kullanıcının sorusunu **detaylı ve mantıklı** bir şekilde yanıtla.
Yanıtın, hukuki ve kesin olmalıdır. Eğer bağlam, soruyu cevaplamak için yeterli bilgiyi içermiyorsa, açıkça "Verilen hukuki metinlerde bu konuda yeterli bilgi bulunmamaktadır." diye cevap ver.

Bağlam:
{context}

Soru: {question}
"""
RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE, 
    input_variables=["context", "question"]
)

# st.cache_resource: Uygulama yeniden yüklense bile RAG bileşenlerini bellekte tutar ve tekrar yüklenmesini engeller.
@st.cache_resource 
def setup_rag_pipeline():
    try:
        GEMINI_API_KEY_VALUE = os.getenv('GEMINI_API_KEY')

        # 2. Modelleri Tanımlama: Cevap üretimi ve kategorizasyon için LLM, vektörleme için Embedding modeli tanımlanır.
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", # Ana cevap üretimi için düşük gecikmeli model.
            temperature=0.2, # Cevabın biraz yaratıcı ama odaklanmış olması için düşük sıcaklık.
            google_api_key=GEMINI_API_KEY_VALUE
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004", # Vektör veritabanı için kullanılan Embedding modeli.
            google_api_key=GEMINI_API_KEY_VALUE
        )
        
        # 3. KAYITLI VERİTABANINI YÜKLEME: Önceden oluşturulmuş Chroma vektör veritabanını yükler.
        vectorstore = Chroma(
            persist_directory="./chroma_db", # Vektörlerin depolandığı dizin.
            embedding_function=embeddings
        )

        llm_categorizer = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0, # Kategorizasyonun kesin olması için sıfır sıcaklık.
            google_api_key=GEMINI_API_KEY_VALUE
        )

        return llm, embeddings, vectorstore, llm_categorizer

    except Exception as e:
        st.error(f"RAG Pipeline Yüklenirken Kritik Hata Oluştu: {e}")
        st.caption("Lütfen 'chroma_db' klasörünün varlığını ve API anahtarının geçerliliğini kontrol edin.")
        return None, None, None, None

# Pipeline bileşenlerini yükle ve hata varsa durdur.
llm, embeddings, vectorstore, llm_categorizer = setup_rag_pipeline()
if llm is None:
    st.stop()

# =================================================================
# 2. STREAMLIT ARAYÜZÜ (Kullanıcı Etkileşimi)
# =================================================================

# Streamlit sayfa ayarları ve başlıkları.
st.set_page_config(page_title="Aile Hukuku Asistanı", layout="centered")
st.title("⚖️ Aile Hukuku Asistanı: TMK RAG Chatbot")
st.caption("Bu uygulama, Türk Medeni Kanunu'nun Aile Hukuku hükümlerine odaklanarak cevaplar üretir. (Gemini Destekli)")

# Kullanıcı Girişi (Metin Kutusu)
user_query = st.text_input("Hukuki sorunuzu girin:", 
                             placeholder="Örn: Türk Medeni Kanunu'na göre, eşlerden birinin diğerini terk etmesi durumunda boşanma davası açılabilmesi için hangi şartlar aranır?",
                             key="user_input_box")

# Sorguyu İşleme: Kullanıcı bir sorgu girdiğinde çalışır.
if user_query:
    with st.spinner("Cevap Aranıyor... (Gemini API çağrısı yapılıyor)"):
        try:
            # ADIM 1: SORGUNUN KAYNAĞINI BELİRLE (Source Routing)
            determined_source = get_source_filter(llm_categorizer, user_query)
            
            # Filtre koşulunu ayarla: Eğer kategori 'Diğer' değilse veya belirlenmişse, veritabanı sorgusuna filtre eklenir.
            filter_condition = {}
            if determined_source and determined_source != "Diğer":
                filter_condition = {"source": determined_source}
                st.info(f"Filtre Uygulandı: Sorgu, **{determined_source}** kaynağına yönlendirildi.")
            else:
                st.info("Filtre Uygulanmadı: Geniş alanda arama yapılıyor.")

            # ADIM 2: RETRIEVER'ı Filtre, MMR ve MultiQuery ile Kurma (Akıllı Veri Çekme)
            
            # 2.1. Temel Retriever'ı oluştur (MMR ve Filtreleme)
            # MMR (Maximal Marginal Relevance): Hem alaka düzeyini hem de çeşitliliği artırarak alakasız tekrarlı sonuçları azaltır.
            base_retriever = vectorstore.as_retriever(
                search_type="mmr",       
                search_kwargs={"k": 5, "fetch_k": 30, "filter": filter_condition} # k=5: 5 sonuç döndür, fetch_k=30: En alakalı 30 sonuç arasından MMR seçimi yap.
            )

            # 2.2. MultiQuery ile Retriever'ı Güçlendirme
            # MultiQueryRetriever: Tek bir kullanıcı sorgusundan, LLM kullanarak birden çok arama sorgusu üretir.
            # Bu, retrieval sonuçlarını zenginleştirerek tek bir sorgunun başarısız olma ihtimalini düşürür.
            retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever, # Temel MMR + Filtreli retriever kullanılır.
                llm=llm_categorizer, # Yeni sorgular üretmek için LLM kullanılır.
            )
            
            # ADIM 3: RAG Zincirini Kurma ve Çalıştırma (Cevap Oluşturma)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, # Ana cevap üretici LLM.
                chain_type="stuff", # Çekilen tüm dokümanları tek bir prompt'a sıkıştırıp LLM'e gönderme yöntemi.
                retriever=retriever, # MultiQuery ve MMR özellikli retriever kullanılır.
                chain_type_kwargs={"prompt": RAG_PROMPT}, # Önceden tanımlanmış gelişmiş prompt şablonu kullanılır.
                return_source_documents=True # Cevapla birlikte kaynak dokümanları da geri döndürmeyi sağlar.
            )
            
            # Zinciri çalıştırma
            result = qa_chain.invoke({"query": user_query})
            cevap = result['result']
            kaynaklar = result['source_documents']

            # --- CEVAP BÖLÜMÜ (Kullanıcıya Gösterim)---
            st.subheader("Chatbot Cevabı")
            st.info(cevap)

            # --- ALINTILAR BÖLÜMÜ (Şeffaflık)---
            st.subheader("Kullanılan Kaynaklar (RAG Alıntıları)")
            if not kaynaklar:
                st.warning("Bu sorgu için hukuki metinlerde alakalı bir kaynak bulunamadı.")
            else:
                # Kullanılan her bir kaynak metni ve kaynağın adını (metadata'dan) gösterir.
                for i, doc in enumerate(kaynaklar):
                    source_name = doc.metadata.get('source', 'Bilinmiyor')
                    st.markdown(f"**{i+1}. Kaynak:** `{source_name}`")
                    st.caption(f"**Bağlam:** {doc.page_content}")
            
        except Exception as e:
            st.error(f"Sorgu sırasında bir hata oluştu: {e}")