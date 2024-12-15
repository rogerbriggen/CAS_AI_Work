# Excercise

* Use Case Definieren
  * Bereit für das Projekt unter <https://docs.google.com/presentation/d/1OLBNGVrBs_uzSdMy5gNh6WnhYh4jheWBwH-1hCASzk8/edit?usp=sharing>
* Dementsprechend eigene PDFs
  * Habe ich  [data\example_pdfs_securiton_en](data\example_pdfs_securiton\FidesNet_PI_en.pdf)
* TPM (token per minute) Limit Problem angehen
  * Gemacht, siehe[openai_ratelimited.py](openai_ratelimited.py)
* Text chunking
  * <https://www.geeksforgeeks.org/how-to-chunk-text-data-a-comparative-analysis/>
  * <https://www.pinecone.io/learn/chunking-strategies/>
* Text cleaning
  * <https://medium.com/intel-tech/four-data-cleaning-techniques-to-improve-large-language-model-llm-performance-77bee9003625>
* Embeddings - hugging face vs OpenAI
  * siehe [embeddings.md](embeddings.md)
* System-prompt und User-Prompt (für den 2 Fall wo man am Ende Fragen an GPT schickt)
  * ?
* Parameter variieren (temperature=0, top_p=0.1)
* Mit RAG ohne RAG vergleich
  * ca. 80% mit Inhalt, nachdem alle 4 Dokumente eingelesen wurden
* Fazit (und was kann man besser machen - wie z.B. Datenbank etc)
  * Mehr Content
  * Bessere embeddings
  * Besser korrelieren
