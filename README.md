# AmbedkarGPT – Intern Task

A minimal **Retrieval-Augmented Generation (RAG)** command-line Q&A system that uses the provided _speech.txt_ and answers user questions strictly based on its content.  
This project is built as part of the **Phase 1 – Core Skills Evaluation** for the AI Intern assignment.

## 📌 Features

- Loads and processes `speech.txt`
- Splits text into chunks
- Generates embeddings using  
  **sentence-transformers/all-MiniLM-L6-v2**
- Stores embeddings locally using **ChromaDB**
- Retrieves relevant chunks for any user question
- Generates grounded answers using **Ollama (Mistral 7B)** locally
- Interactive command-line interface

Everything runs **offline** — no API keys, no accounts, no costs.

## 📂 Project Structure

```
AmbedkarGPT-Intern-Task/
│
├── main.py
├── speech.txt
├── requirements.txt
└── README.md
```

## 🛠️ Installation & Setup

### 1. Clone this repository

```bash
git clone https://github.com/<your-username>/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### 2. Create and activate a virtual environment

#### macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🤖 Setting Up Ollama (Required)

### 1. Install Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Or download from: https://ollama.ai

### 2. Pull the Mistral 7B model

```bash
ollama pull mistral
```

### 3. Verify installation

```bash
ollama list
```

## ▶️ How to Run the Project

### First run (build vector DB + start Q&A)

```bash
python main.py
```

### Force rebuild the vector store

```bash
python main.py --rebuild
```

## 💬 Interacting With the System

You'll see:

```
AmbedkarGPT — ask questions about the speech. Type 'exit' to quit.
```

Example queries:

```
Q: What is the real remedy according to the speech?
Q: What is the role of the shastras?
Q: Why does the speaker criticize social reform?
```

Type `exit` to quit.

## 📦 Requirements

- Python 3.8+
- Ollama installed locally
- Hardware capable of running Mistral 7B

## 📘 About the Data

`speech.txt` contains the assignment-provided excerpt from **“Annihilation of Caste” — Dr. B.R. Ambedkar**.  
The system answers _only_ using this text.

## ⚙️ Pipeline Overview

1. Load text (`TextLoader`)
2. Split into chunks (`CharacterTextSplitter`)
3. Create embeddings (`all-MiniLM-L6-v2`)
4. Store vectors locally (ChromaDB)
5. Retrieve using `.invoke(query)`
6. Generate answer via **Ollama Mistral**

## 📝 Notes

- Runs fully offline
- If you modify speech.txt, run:

```bash
python main.py --rebuild
```

## 📨 Submission

This repo fulfills the **Kalpit Pvt Ltd – AI Intern Hiring** Phase 1 requirements.