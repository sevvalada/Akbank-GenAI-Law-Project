import streamlit as st
import os

# LangChain Çekirdek Paketler
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# HATA ÇÖZÜMÜ: Zincirleri, en güvenilir adres olan 'langchain_community.chains'den çekiyoruz.
# Eğer 'from langchain.chains import X' hata veriyorsa, bu yolla çözülmelidir.
from langchain_community.chains import RetrievalQA
from langchain_community.chains import LLMChain

from langchain.prompts import PromptTemplate
from typing import Dict, Any
from langchain.retrievers.multi_query import MultiQueryRetriever # MultiQuery için gerekli

# =================================================================
# YARDIMCI FONKSİYON: SORGUYU KATEGORİZE ETME
# =================================================================
def get_source_filter(llm_categorizer: ChatGoogleGenerativeAI, query: str) -> str:
    """
    Gemini modelini kullanarak kullanıcı sorgusunun ait olduğu hukuki kaynağı belirler.
    Bu, Retrieval alanını daraltır.
    """
    source_list = ["Türkiye Cumhuriyeti Anayasası", "Türk Medeni Kanunu", "Türk Ceza Kanunu", "Borçlar Kanunu", "İcra ve İflas Kanunu", "Diğer"]
    source_prompt = PromptTemplate(
        template=f"""Kullanıcı sorgusunu analiz et ve sorgunun en çok hangi hukuki kaynağa ait olduğunu belirle.
        Sadece aşağıdaki kaynaklardan birini cevap olarak döndür: {', '.join(source_list)}. 
        Eğer sorgu bu kaynaklardan birine ait değilse, 'Diğer' olarak cevap ver. 
        Sorgu: {{query}}
        Kaynak:""",
        input_variables=["query"]
    )
    
    chain = LLMChain(llm=llm_categorizer, prompt=source_prompt)
    
    try:
        response = chain.run(query=query).strip()
        if response in source_list:
            return response
        else:
            return None
    except Exception:
        return None 

# =================================================================
# 1. ORTAM KONTROLÜ VE RAG PIPELINE YÜKLEMESİ
# =================================================================

if 'GEMINI_API_KEY' not in os.environ:
    st.error("HATA: GEMINI_API_KEY ortam değişkeni tanımlı değil. Lütfen terminalde ayarlayın (export/set).")
    st.stop()

# RAG Prompt Şablonu (Gelişmiş Yorumlama)
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

@st.cache_resource 
def setup_rag_pipeline():
    try:
        GEMINI_API_KEY_VALUE = os.getenv('GEMINI_API_KEY')

        # 2. Modelleri Tanımlama
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=GEMINI_API_KEY_VALUE
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            google_api_key=GEMINI_API_KEY_VALUE
        )
        
        # 3. KAYITLI VERİTABANINI YÜKLEME
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )

        llm_categorizer = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            google_api_key=GEMINI_API_KEY_VALUE
        )

        return llm, embeddings, vectorstore, llm_categorizer

    except Exception as e:
        st.error(f"RAG Pipeline Yüklenirken Kritik Hata Oluştu: {e}")
        st.caption("Lütfen 'chroma_db' klasörünün varlığını ve API anahtarının geçerliliğini kontrol edin.")
        return None, None, None, None

llm, embeddings, vectorstore, llm_categorizer = setup_rag_pipeline()
if llm is None:
    st.stop()

# =================================================================
# 2. STREAMLIT ARAYÜZÜ (İLK HALİ)
# =================================================================

# Layout ayarı (Streamlit'in varsayılan ayarları)
st.set_page_config(page_title="Aile Hukuku Asistanı", layout="centered")

st.title("⚖️ Aile Hukuku Asistanı: TMK RAG Chatbot")
st.caption("Bu uygulama, Türk Medeni Kanunu'nun Aile Hukuku hükümlerine odaklanarak cevaplar üretir. (Gemini Destekli)")

# Kullanıcı Girişi
user_query = st.text_input("Hukuki sorunuzu girin:", 
                            placeholder="Örn: Türk Medeni Kanunu'na göre, eşlerden birinin diğerini terk etmesi durumunda boşanma davası açılabilmesi için hangi şartlar aranır?",
                            key="user_input_box")

# Sorguyu İşleme
if user_query:
    with st.spinner("Cevap Aranıyor... (Gemini API çağrısı yapılıyor)"):
        try:
            # ADIM 1: SORGUNUN KAYNAĞINI BELİRLE
            determined_source = get_source_filter(llm_categorizer, user_query)
            
            # Filtre koşulunu ayarla
            filter_condition = {}
            if determined_source and determined_source != "Diğer":
                filter_condition = {"source": determined_source}
                st.info(f"Filtre Uygulandı: Sorgu, **{determined_source}** kaynağına yönlendirildi.")
            else:
                st.info("Filtre Uygulanmadı: Geniş alanda arama yapılıyor.")

            # ADIM 2: RETRIEVER'ı Filtre, MMR ve MultiQuery ile Kurma
            
            # 2.1. Temel Retriever'ı oluştur (MMR ve Filtreleme)
            base_retriever = vectorstore.as_retriever(
                search_type="mmr",       
                search_kwargs={"k": 5, "fetch_k": 30, "filter": filter_condition}
            )

            # 2.2. MultiQuery ile Retriever'ı Güçlendirme
            retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm_categorizer,
            )
            
            # ADIM 3: RAG Zincirini Kurma ve Çalıştırma (Gelişmiş Prompt Kullanılıyor)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": RAG_PROMPT}, 
                return_source_documents=True
            )
            
            result = qa_chain.invoke({"query": user_query})
            cevap = result['result']
            kaynaklar = result['source_documents']

            # --- CEVAP BÖLÜMÜ ---
            st.subheader("Chatbot Cevabı")
            st.info(cevap)

            # --- ALINTILAR BÖLÜMÜ ---
            st.subheader("Kullanılan Kaynaklar (RAG Alıntıları)")
            if not kaynaklar:
                st.warning("Bu sorgu için hukuki metinlerde alakalı bir kaynak bulunamadı.")
            else:
                for i, doc in enumerate(kaynaklar):
                    source_name = doc.metadata.get('source', 'Bilinmiyor')
                    st.markdown(f"**{i+1}. Kaynak:** `{source_name}`")
                    # Metin kesilmeyecek, tam olarak gösterilecek
                    st.caption(f"**Bağlam:** {doc.page_content}")
            
        except Exception as e:
            st.error(f"Sorgu sırasında bir hata oluştu: {e}")