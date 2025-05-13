# FishAI - Piattaforma Integrata per il Monitoraggio Ittico

**FishAI** è una piattaforma avanzata per il riconoscimento delle specie ittiche, l’analisi degli ecosistemi acquatici e la gestione automatizzata dei dati tramite intelligenza artificiale e interfacce interattive.

---

## 🧩 Componenti Principali

1. **FishAI GUI (basata su Gradio)**  
   Interfaccia web principale per l’analisi visiva, la generazione di report e l’interazione con l’AI.  
   _(Questo repository contiene il codice relativo a questa componente.)_

2. **Fishbot (basato su n8n)**  
   Sistema di automazione che consente di interagire via bot (es. Telegram) per l’analisi media e la generazione automatica di risposte e report.  
   _(Il codice può risiedere in un altro repository o essere descritto concettualmente qui.)_

3. **Backend di Analisi e AI**  
   Moduli Python per riconoscimento visivo (es. TensorFlow), interazione con AI conversazionale (es. OpenAI GPT-4o), e gestione dati. Utilizzati sia dalla GUI sia, potenzialmente, dal bot tramite API o chiamate di sistema.

---

## ⚙️ Funzionalità Chiave

### 🎯 Riconoscimento Specie

- Analisi di **immagini** (.jpg, .jpeg, .heic/heif) con **estrazione automatica dei metadati GPS (EXIF)**.
- Analisi di **video** locali e **stream da URL** (YouTube, RTSP, HTTP).
- Supporto per **analisi in tempo reale** tramite webcam collegata.

### 🤖 Interazione AI Avanzata

- **Chatbot visivo**: dialogo con un’AI via testo e immagini, con risposte su specie, habitat e comportamento.
- **Assistenza nella redazione di report**: riassunto, correzione o commento di testi Markdown tramite AI.
- **Generazione automatica di grafici**: basta descriverlo e l’AI genera un grafico dai dati di rilevamento.

### 📊 Gestione e Visualizzazione Dati

- **Log centralizzato**: archivia automaticamente rilevamenti con timestamp, specie, probabilità e coordinate (da EXIF o stimate).
- **Editor report avanzato**: supporto Markdown, integrazione immagini e grafici AI, esportazione in HTML, PDF o Word.
- **Grafici standalone**: creazione di heatmap temporali, barre, scatter, ecc.
- **Mappa habitat interattiva**: visualizzazione su mappa (Folium) dei luoghi di rilevamento con layer differenziati (EXIF vs AI).

### 🔁 Integrazione Automatizzata (Fishbot)

- Permette di **inviare input multimediali** o **domande** via bot (es. Telegram).
- Attiva flussi automatici (workflow n8n) per analisi, risposta AI e salvataggio dei risultati nella piattaforma.

---

## 🛠️ Tecnologie Utilizzate

- **Gradio**: per la GUI interattiva.
- **TensorFlow / Keras**: per il riconoscimento visivo.
- **OpenAI GPT-4o**: per le risposte intelligenti e la generazione contenuti.
- **Pillow + pillow-heif**: per la gestione delle immagini e metadati EXIF.
- **OpenCV**: per l’analisi dei frame video.
- **Matplotlib**: per la generazione grafica.
- **Folium**: per la visualizzazione geografica interattiva.
- **yt-dlp**: per l’estrazione di flussi video (es. da YouTube).
- **markdown2, pdfkit, pandoc**: per l’esportazione di report.
- **dotenv**: per la gestione delle variabili ambientali.
- **n8n**: per la componente di automazione Fishbot.

---

## 🚀 Installazione (GUI FishAI)

1. **Clonare il repository:**

```bash
git clone https://github.com/NOME_UTENTE/NOME_REPO.git
cd NOME_REPO
