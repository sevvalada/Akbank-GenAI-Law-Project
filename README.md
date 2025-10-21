# ⚖️ Aile Hukuku Chatbotu: Türk Medeni Kanunu Tabanlı Akıllı Hukuk Asistanı

## Projenin Amacı

Bu proje, **Akbank GenAI Bootcamp** kapsamında geliştirilmiş **Retrieval Augmented Generation (RAG)** tabanlı bir yapay zekâ projesidir.  
**Aile Hukuku Chatbotu**, Türk Medeni Kanunu’nun **Aile Hukuku** hükümlerine dayanarak, kullanıcıların hukuki sorularına doğru, hızlı ve yorum içermeyen cevaplar sunmayı amaçlar.  

Bu sistem, özellikle **boşanma, evlilik, velayet, mal rejimi ve nafaka** konularında kullanıcıya destek sağlar.

---

## Veri Seti Hakkında

Proje, **Türk Medeni Kanunu’nun Aile Hukuku bölümlerinden** oluşturulmuş özel bir veri seti kullanmaktadır.  
Veri seti, aşağıdaki ana başlıklardan alınmıştır:

- Evliliğin Şartları ve Hükümleri  
- Boşanma Nedenleri ve Sonuçları  
- Eşlerin Hak ve Yükümlülükleri  

> **NOT:** Proje gereklilikleri doğrultusunda, veri seti dosyası (`turkish_law_dataset.csv`) GitHub deposuna yüklenmemiştir.

---

## Çözüm Mimarisi

Proje, **LangChain** ve **Google Gemini API** teknolojileriyle desteklenen bir **RAG mimarisi** üzerine kurulmuştur.  
Bu yapı, kullanıcıdan gelen soruları analiz eder, uygun yasa maddelerini bulur ve güvenilir, kaynaklı cevap üretir.

### Kullanılan Teknolojiler

| Bileşen | Teknoloji | Amaç |
| :--- | :--- | :--- |
| Generation Model | Google Gemini 2.5 Flash | Nihai cevabı üretir. |
| Embedding Model | Google text-embedding-004 | Metinleri vektör uzayına dönüştürür. |
| Vektör Veritabanı | ChromaDB | Aile Hukuku metinlerini kalıcı olarak saklar. |
| RAG Framework | LangChain | Belgeleri çağırır ve LLM ile entegre eder. |
| Web Arayüzü | Streamlit | Kullanıcıya etkileşimli sohbet arayüzü sunar. |

### Gelişmiş Retrieval Katmanı

1. **Source Routing (Kaynak Yönlendirme):** Sorgular, LLM tarafından analiz edilerek uygun kaynak (TMK, Anayasa vb.) yönlendirmesi yapılır.  
2. **MultiQuery Retriever:** Kullanıcı sorgusundan türetilen çoklu arama sorguları ile daha geniş bağlam yakalanır.  
3. **MMR (Maximal Marginal Relevance):** Alaka düzeyini maksimize ederken, dönen belgelerin çeşitliliğini korur.  

---

## Elde Edilen Sonuçlar

Geliştirilen sistem, **hukuki bilgiye erişim süresini kısaltmış** ve **halüsinasyon riskini önemli ölçüde azaltmıştır.**  
Cevaplar, kaynak gösterimiyle birlikte sunularak şeffaf bir yapay zekâ deneyimi sağlamaktadır.

---

## Web Arayüzü

**Canlı Uygulama Linki:**  
https://akbank-genai-law-project-3rddmmjewhhcgx6rhwxikq.streamlit.app/

Bu bağlantı üzerinden chatbot arayüzüne erişebilir, Aile Hukuku kapsamında yasal sorularınızı test edebilirsiniz.  
Uygulama, her cevabın dayandığı **TMK maddelerini** kullanıcıya şeffaf biçimde sunar.

---

## Örnek Sohbet

**Kullanıcı:**  
> Eşlerden birinin akıl sağlığının bozulması durumunda boşanma davası açılabilir mi?

**Aile Hukuku Chatbotu:**  
> Türk Medeni Kanunu madde 165'e göre, eşlerden biri akıl hastası olup, bu hastalık evlilik birliğini diğer eş için çekilmez hale getirmişse, boşanma davası açılabilir.  
> Bu durumda hastalığın geçmeyeceğine dair sağlık kurulu raporu gerekir.  

**Kaynak:** TMK m.165


---
## Uygulama Görünümü

| Görsel | Açıklama |
| :--- | :--- |
| <img width="1046" height="517" alt="image" src="https://github.com/user-attachments/assets/bad9d82f-9be0-44c1-9c8d-3043239aecd8" /> | Ana Sohbet Arayüzü |
| <img width="869" height="805" alt="Yasa Yönlendirme" src="https://github.com/user-attachments/assets/289001a4-1318-4354-8ac3-49410fa56aba" /> | TMK Maddesi Gösterimi |
| <img width="975" height="716" alt="Sonuç ve Kaynak" src="https://github.com/user-attachments/assets/c6945002-c450-4aa9-96dc-b171d8e7e18e" /> | Sonuç ve Kaynak Gösterimi |


