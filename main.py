#!/usr/bin/env python3
"""
AmbedkarGPT - RAG prototype without importing langchain.chains,
to avoid compatibility issues with recent LangChain package splits.

Works with:
- langchain_community.document_loaders.TextLoader
- langchain_text_splitters.CharacterTextSplitter
- langchain_community.embeddings.HuggingFaceEmbeddings
- langchain_community.vectorstores.Chroma
- langchain_ollama.Ollama

Behavior:
- Build or load a local Chroma vector store of embeddings (sentence-transformers/all-MiniLM-L6-v2)
- Retrieve top-k chunks for each user question
- Call local Ollama (mistral) to generate an answer constrained to retrieved context
"""

import argparse
import os
from typing import List

# updated import locations (avoid legacy langchain.* imports)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM as Ollama


def build_vectorstore(speech_path: str, persist_directory: str = "chroma_db") -> Chroma:
    print(f"[+] Loading text from: {speech_path}")
    loader = TextLoader(speech_path, encoding="utf-8")
    docs = loader.load()

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
    )
    print("[+] Splitting documents into chunks...")
    docs_split = splitter.split_documents(docs)
    print(f"[+] Number of chunks: {len(docs_split)}")

    print("[+] Initializing HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"[+] Creating Chroma vectorstore (persist_directory='{persist_directory}')...")
    vectordb = Chroma.from_documents(
        documents=docs_split,
        embedding=hf,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print("[+] Vectorstore persisted.")
    return vectordb


def load_vectorstore(persist_directory: str = "chroma_db") -> Chroma:
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Persist directory '{persist_directory}' not found. Run with --rebuild.")
    print(f"[+] Loading Chroma from '{persist_directory}'")
    hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=hf)
    return vectordb


def retrieve_documents(vectordb: Chroma, query: str, top_k: int = 4) -> List[str]:
    """
    Use the vectorstore's retriever to fetch relevant documents. Return list of page_content strings.
    """
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    docs = retriever.invoke(query)
    contents = [getattr(d, "page_content", str(d)) for d in docs]
    return contents


def generate_answer_with_ollama(ollama_model: str, context_chunks: List[str], question: str) -> str:
    """
    Build a prompt that instructs the model to use ONLY provided context.
    Calls Ollama via langchain_ollama wrapper.
    """
    # join context neatly (numbered)
    numbered_context = "\n\n".join(f"[{i+1}] {chunk.strip()}" for i, chunk in enumerate(context_chunks))
    prompt = (
        "You are an assistant that must answer questions using ONLY the provided context.\n"
        "If the answer is not contained in the context, respond: \"I don't know based on the provided text.\"\n"
        "Be concise and factual.\n\n"
        f"Context:\n{numbered_context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    # instantiate Ollama LLM (talks to local ollama daemon). Ensure `ollama pull mistral` was run.
    llm = Ollama(model=ollama_model, temperature=0.0, max_tokens=512)

    # Call the LLM. langchain_ollama's LLM follows the LangChain interface: llm(prompt) -> str
    resp = llm.invoke(prompt)

# New Ollama output is a dict: {'id':..., 'object':..., 'model':..., 'output_text':...}
    if isinstance(resp, dict):
        # Newer keys: 'message', 'output_text', 'text'
        if "output_text" in resp:
            return resp["output_text"].strip()
        if "text" in resp:
            return resp["text"].strip()
        if "message" in resp and "content" in resp["message"]:
            return resp["message"]["content"].strip()

    # Fallback
    return str(resp).strip()


def interactive_loop(vectordb: Chroma, ollama_model: str, top_k: int = 4):
    print("\nAmbedkarGPT — ask questions about the speech. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            q = input("Q: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        try:
            chunks = retrieve_documents(vectordb, q, top_k=top_k)
            if not chunks:
                print("No relevant context found in the vector store. Try re-building the vectors or rephrase.")
                continue

            answer = generate_answer_with_ollama(ollama_model, chunks, q)
            print("\nA:", answer, "\n")
            print("---- Retrieved chunks (for debugging) ----")
            for i, c in enumerate(chunks, 1):
                snippet = c.replace("\n", " ").strip()
                if len(snippet) > 400:
                    snippet = snippet[:400] + "..."
                print(f"[{i}] {snippet}\n")
            print("------------------------------------------\n")
        except Exception as exc:
            print("Error while answering:", repr(exc))


def main():
    parser = argparse.ArgumentParser(description="AmbedkarGPT (langchain-compat-safe version)")
    parser.add_argument("--speech", default="speech.txt", help="Path to speech.txt")
    parser.add_argument("--persist-dir", default="chroma_db", help="Chroma persistence directory")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild vectorstore")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--ollama-model", default="mistral", help="Ollama model name (default 'mistral')")
    args = parser.parse_args()

    if not os.path.exists(args.speech):
        print(f"[!] Cannot find speech file at '{args.speech}'. Please add speech.txt and retry.")
        return

    if args.rebuild or not os.path.exists(args.persist_dir):
        vectordb = build_vectorstore(args.speech, args.persist_dir)
    else:
        vectordb = load_vectorstore(args.persist_dir)

    interactive_loop(vectordb, args.ollama_model, top_k=args.top_k)


if __name__ == "__main__":
    main()