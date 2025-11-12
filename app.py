from dotenv import load_dotenv
import os
import re
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import gradio as gr

# ====== OpenAI only ======
from openai import OpenAI


def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length for cosine similarity via inner product."""
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norm


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> List[str]:
    """
    Naive character-length chunking with light sentence-aware splitting
    for both Chinese and English punctuation.
    """
    if not text:
        return []
    # Collapse whitespace
    t = re.sub(r'\s+', ' ', text).strip()

    # First, coarse split on sentence-ending punctuation (CN/EN)
    sentences = re.split(r'(?<=[。！？.!?])\s+', t)
    out = []
    buf = ""

    for s in sentences:
        if len(buf) + len(s) + 1 <= chunk_size:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                out.append(buf)
            # If a single sentence is very long, further split by length
            if len(s) > chunk_size:
                for i in range(0, len(s), chunk_size - chunk_overlap):
                    out.append(s[i:i + chunk_size])
                buf = ""
            else:
                buf = s
    if buf:
        out.append(buf)

    # Add overlap across adjacent chunks to reduce context fragmentation
    if chunk_overlap > 0 and len(out) >= 2:
        merged = []
        for i, seg in enumerate(out):
            if i == 0:
                merged.append(seg)
            else:
                prev = merged[-1]
                overlap = prev[-chunk_overlap:] if len(prev) >= chunk_overlap else prev
                merged.append((overlap + " " + seg).strip())
        out = merged
    return out


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    order: int


class RAGIndex:
    """Holds the embedding model, FAISS index, and chunk metadata."""
    def __init__(self, embed_model: str = "moka-ai/m3e-base"):
        self.embed_model_name = embed_model
        self._embedder = SentenceTransformer(self.embed_model_name)
        self.dim = self._embedder.get_sentence_embedding_dimension()
        # Inner product on unit vectors == cosine similarity
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks: List[Chunk] = []
        self.matrix = None  # optional dense matrix cache (not required for search)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Encode texts into L2-normalized float32 vectors."""
        vecs = self._embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return _normalize(vecs.astype("float32"))

    def add_chunks(self, chunks: List[Chunk]):
        """Add chunks to metadata and FAISS index."""
        self.chunks.extend(chunks)
        vectors = self.embed([c.text for c in chunks])
        if self.matrix is None:
            self.matrix = vectors
        else:
            self.matrix = np.vstack([self.matrix, vectors])
        self.index.add(vectors)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Search top_k most similar chunks for a query."""
        q = self.embed([query])
        scores, ids = self.index.search(q, top_k)
        result = []
        for idx, score in zip(ids[0].tolist(), scores[0].tolist()):
            if idx == -1:
                continue
            result.append((self.chunks[idx], float(score)))
        return result

    # Persist / restore index (faiss + metadata)
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        meta = [asdict(c) for c in self.chunks]
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"embed_model": self.embed_model_name, "chunks": meta}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "RAGIndex":
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(embed_model=meta.get("embed_model", "moka-ai/m3e-base"))
        obj.index = faiss.read_index(os.path.join(path, "index.faiss"))
        obj.chunks = [Chunk(**c) for c in meta["chunks"]]
        obj.dim = obj.index.d
        # Rebuilding self.matrix is optional; FAISS can search without it.
        return obj


def read_pdf_chunks(file_path: str, chunk_size: int = 500, chunk_overlap: int = 80) -> List[Chunk]:
    """Extract text from a PDF and split it into Chunk objects with metadata."""
    reader = PdfReader(file_path)
    chunks: List[Chunk] = []
    order = 0
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if not text.strip():
            continue
        segs = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for s in segs:
            chunks.append(Chunk(text=s, source=os.path.basename(file_path), page=i + 1, order=order))
            order += 1
    return chunks


def build_prompt(contexts: List[Tuple[Chunk, float]], question: str) -> str:
    """Construct a strict, citation-required prompt from retrieved contexts."""
    bullet = []
    for i, (ck, score) in enumerate(contexts, 1):
        bullet.append(f"[{i}] (p.{ck.page} | {ck.source}) {ck.text}")
    context_block = "\n\n".join(bullet)
    system = (
        "You are a careful document QA assistant. Answer ONLY using the 'Provided Materials'. "
        "You MUST append citation markers with the chunk number and page like [1,p.3]. "
        "If the materials are insufficient to answer, reply: Cannot be determined from the provided materials."
    )
    user = (
        f"Provided Materials:\n{context_block}\n\n"
        f"Question: {question}\n"
        f"Please answer concisely and append citation numbers at the end."
    )
    return system, user


def generate_with_openai(system: str, user: str, model: str = None) -> str:
    """Call OpenAI Chat Completions API."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    # simple retries for resilience in class demos
    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenAI generation failed after retries: {last_err}")


def answer_question(rag: RAGIndex, question: str, top_k: int = 5) -> Dict:
    """Full RAG pipeline for one question: retrieve → prompt → generate → package outputs."""
    contexts = rag.search(question, top_k=top_k)
    sys_prompt, user_prompt = build_prompt(contexts, question)

    try:
        answer = generate_with_openai(sys_prompt, user_prompt)
    except Exception as e:
        answer = f"Generation failed: {e}"

    cites = [f"[{i+1},p.{ck.page}]" for i, (ck, _) in enumerate(contexts)]
    retrieved = [
        {"rank": i+1, "score": round(float(score), 4), "page": ck.page, "source": ck.source, "text": ck.text}
        for i, (ck, score) in enumerate(contexts)
    ]
    return {"answer": answer, "citations": cites, "retrieved": retrieved}



def ui_build_index(files, chunk_size, chunk_overlap, embed_model):
    """Callback to build an index from uploaded PDFs."""
    if not files:
        return gr.update(value="Please upload PDF files first."), None

    rag = RAGIndex(embed_model=embed_model)
    total_chunks = 0
    for f in files:
        chunks = read_pdf_chunks(f.name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        total_chunks += len(chunks)
        if len(chunks) == 0:
            continue
        rag.add_chunks(chunks)

    # Save to a temp directory (demo)
    save_dir = "./_rag_index"
    rag.save(save_dir)
    msg = (
        "Index built \n"
        f"- Files: {len(files)}\n"
        f"- Chunks: {total_chunks}\n"
        f"- Embedding model: {embed_model}\n"
        f"- Index dir: {save_dir}"
    )
    return msg, rag


def ui_chat(question, rag: RAGIndex, top_k):
    """Callback to answer a question using the currently built/loaded index."""
    if not question or not rag:
        return "Please build the index first.", ""
    out = answer_question(rag, question, top_k=top_k)
    # Show retrieved chunks with scores
    lines = []
    for r in out["retrieved"]:
        lines.append(f'[{r["rank"]}] (score={r["score"]}, p.{r["page"]} | {r["source"]}) {r["text"]}')
    retrieved_block = "\n\n".join(lines)
    return out["answer"], retrieved_block


def build_demo():
    with gr.Blocks(title="Document QA Bot (RAG)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 📄 Document QA Bot (RAG)\nUpload PDFs → Build a vector index → Ask questions and get answers with citations.")
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDF(s)")
                chunk_size = gr.Slider(200, 1200, value=500, step=50, label="Chunk Size (max chars per chunk)")
                chunk_overlap = gr.Slider(0, 200, value=80, step=10, label="Chunk Overlap (chars)")
                embed_model = gr.Dropdown(
                    choices=[
                        "moka-ai/m3e-base",
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        "thenlper/gte-large"
                    ],
                    value="moka-ai/m3e-base",
                    label="Embedding Model (recommended: m3e-base)"
                )
                build_btn = gr.Button("Build Index", variant="primary")
                build_info = gr.Textbox(label="Build Info", interactive=False, max_lines=20, lines=10)
            with gr.Column(scale=2):
                qa = gr.State(None)
                question = gr.Textbox(label="Enter your question", placeholder="e.g., What are the main goals of this document?")
                top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K retrieved chunks")
                ask_btn = gr.Button("Ask")
                answer = gr.Markdown("**The answer will appear here**")
                retrieved = gr.Textbox(label="Retrieved chunks (sorted by relevance)", lines = 20, max_lines = 60, interactive=False)

        build_btn.click(ui_build_index, inputs=[files, chunk_size, chunk_overlap, embed_model], outputs=[build_info, qa])
        ask_btn.click(ui_chat, inputs=[question, qa, top_k], outputs=[answer, retrieved])

        gr.Markdown(
            "> Generator: OpenAI."
        )

    return demo


if __name__ == "__main__":
    print(234)
    demo = build_demo()
    print(123)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share = True)
