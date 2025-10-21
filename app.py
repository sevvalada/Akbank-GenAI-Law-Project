import streamlit as st
import os
import langchain
import sys

print("LangChain version:", langchain.__version__)
print("Python version:", sys.version)

from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

# =================================================================
# YARDIMCI FONKSİYON: SORGUYU KATEGORİZE ETME
# =================================================================
def get_source_filter(llm_categorizer: ChatGoogleGenerativeAI, query: str) -> str:
    source_list = [
        "Türkiye Cumhuriyeti Anayasası",
        "Türk Medeni Kanunu",
        "Türk Ceza Kanunu",
        "Borçlar Kanunu",
        "İcra ve İflas Kanunu",
        "Diğer"
    ]
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
# ORTAM KONTROLÜ VE RAG PIPELINE YÜKLEME
# =================================================================
if 'GEMINI_API_KEY' not in os.environ:
    st.error("HATA: GEMINI_API_KEY ortam değişkeni tanımlı değil. Lütfen terminalde ayarlayın (export/set).")
    st.stop()

RAG_PROMPT_TEMPLATE = """
Sen, Türk hukuku metinleri hakkında cevaplar üreten bir asistansın.
Aşağıdaki 'Bağlam' kısmında sana sunulan hukuki metin parçalarını kullanarak, sadece bu metinlere dayanarak (ek bilgi eklemeden) kullanıcının sorusunu **detaylı ve mantıklı** bir şekilde yanıtla.
Yanıtın, hukuki ve kesin olmalıdır. Eğer bağlam, soruyu cevaplamak için yeterli bilgiyi içermiyorsa, açıkça "Verilen hukuki metinlerde bu konuda yeterli bilgi bulunmamaktadır." diye cevap ver.

Bağlam:
{context}

Soru: {question}
"""
RAG_PROMPT = PromptTemplate(template=RAG_PROMPT_TEMPLATE, input_variables=["context", "question"])

@st.cache_resource
def setup_rag_pipeline():
    try:
        GEMINI_API_KEY_VALUE = os.getenv('GEMINI_API_KEY')

        # LLM ve Embedding
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=GEMINI_API_KEY_VALUE
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",  # Buradaki model ismi LangChain 0.0.350 uyumlu
            google_api_key=GEMINI_API_KEY_VALUE
        )

        # Chroma Vectorstore
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )

        # Kaynak kategorisi için LLM
        llm_categorizer = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            google_api_key=GEMINI_API_KEY_VALUE
        )

        return llm, embeddings, vectorstore, llm_categorizer

    except Exception as e:
        st.error(f"RAG Pipeline Yüklenirken Hata: {e}")
        st.caption("Lütfen 'chroma_db' klasörünün varlığını ve API anahtarının geçerliliğini kontrol edin.")
        return None, None, None, None

llm, embeddings, vectorstore, llm_categorizer = setup_rag_pipeline()
if llm is None:
    st.stop()

# =================================================================
# STREAMLIT ARAYÜZÜ
# =================================================================
st.set_page_config(page_title="Aile Hukuku Asistanı", layout="centered")

st.title("⚖️ Aile Hukuku Asistanı: TMK RAG Chatbot")
st.caption("Bu uygulama, Türk Medeni Kanunu'nun Aile Hukuku hükümlerine odaklanarak cevaplar üretir. (Gemini Destekli)")

user_query = st.text_input(
    "Hukuki sorunuzu girin:",
    placeholder="Örn: Eşlerden birinin akıl sağlığının bozulması durumunda boşanma davası açılabilir mi?",
    key="user_input_box"
)

if user_query:
    with st.spinner("Cevap Aranıyor..."):
        try:
            determined_source = get_source_filter(llm_categorizer, user_query)
            filter_condition = {}
        
            st.info("Filtre Uygulanmadı: Geniş alanda arama yapılıyor.")

            base_retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 30, "filter": filter_condition}
            )

            retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm_categorizer
            )

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

            st.subheader("Chatbot Cevabı")
            st.info(cevap)

            st.subheader("Kullanılan Kaynaklar (RAG Alıntıları)")
            if not kaynaklar:
                st.warning("Bu sorgu için hukuki metinlerde alakalı bir kaynak bulunamadı.")
            else:
                for i, doc in enumerate(kaynaklar):
                    source_name = doc.metadata.get('source', 'Bilinmiyor')
                    st.markdown(f"**{i+1}. Kaynak:** `{source_name}`")
                    st.caption(f"**Bağlam:** {doc.page_content}")

        except Exception as e:
            st.error(f"Sorgu sırasında bir hata oluştu: {e}")
