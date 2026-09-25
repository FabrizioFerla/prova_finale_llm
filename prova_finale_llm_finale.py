# Imports
import os
import json
import uuid
import numpy as np
import chromadb
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from email import policy
from email.parser import BytesParser
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Costanti
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 32
TOP_K_RETRIEVAL = 8
MAX_TOKENS = 4096
DEVICE_MAP = "auto"
LOCAL_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
GROQ_MODEL = "openai/gpt-oss-120b"
LLM_API_BASE_URL= "https://api.groq.com/openai/v1"
BASE_FOLDER = Path(__file__).resolve().parent
VECTOR_FOLDER = "./chroma_db"
COLLECTION_NAME = "datatrust_documents"
DATA_FOLDER = BASE_FOLDER / "documenti_fittizi"
OUTPUT_FOLDER = BASE_FOLDER / "output"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".eml"}

THEMES = {
    "compliance": "rischi di compliance, obblighi normativi, scadenze regolamentari",
    "financial_performance": "risultati finanziari, ricavi, margini, indicatori di performance",
    "strategic_directions": "decisioni strategiche, piani futuri, iniziative di investimento",
    "Governance":"struttura organizzativa, ruoli e responsabilità, processi decisionali, controlli interni, deleghe e poteri",
    "Relazioni con i clienti": "relazioni con i clienti, soddisfazione e reclami, fidelizzazione, contratti e accordi commerciali, feedback dal mercato"
}

CATEGORIES = [
    "Compliance",
    "Performance Finanziaria",
    "Strategia Aziendale",
    "Governance",
    "Relazioni con i clienti"
]


# Schemi e Dataclass
class _LLMAnalysisSchema(BaseModel):
    category: str = Field(description="Categoria del documento")
    category_confidence: float = Field(default=0.0, description="Valore numerico tra 0.0 e 1.0")
    summary: str = Field(default="", description="Riassunto in 2-3 frasi")
    key_insights: List[str] = Field(default_factory=list)
    recorded_dates: List[str] = Field(default_factory=list)
    detected_entities: List[str] = Field(default_factory=list)
    detected_amounts: List[str] = Field(default_factory=list)
    critical_issues: List[str] = Field(default_factory=list)
    recommendation: str = Field(default="")


@dataclass
class DocumentChunk:
    """Rappresentazione di un singolo chunk di testo"""
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class DocumentAnalysis:
    """
    Contiene i risultati dell'analisi estratti per un singolo documento
    """
    file_name: str
    format: str
    category: str = "Unclassified"
    category_confidence: float = 0.0
    summary: str = ""
    key_insights: List[str] = field(default_factory=list)
    detected_dates: List[str] = field(default_factory=list)
    detected_amounts: List[str] = field(default_factory=list)
    detected_entities: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    recommendation: str = ""
    number_of_chunks_analyzed: int = 0

# Estrattori
class Extractor(ABC):
    """Classe base astratta per la lettura dei documenti."""
    format: str  # Formato del documento

    @abstractmethod
    def extract(self, file_path: Path) -> List[Document]:
        """Metodo astratto per estrarre il contenuto in base al formato del file."""
        ...

    def _enrich_metadata(self, file_path: Path, documents: List[Document]) -> List[Document]:
        """Arricchisce i metadati del documento aggiungendo nome file e formato."""
        for doc in documents:
            doc.metadata["file_name"] = file_path.name
            doc.metadata["format"] = self.format
        return documents


class PDFExtractor(Extractor):
    """Estrae il contenuto di un file PDF tramite PyPDFLoader di LangChain."""
    format = "PDF"

    def extract(self, file_path: Path) -> List[Document]:
        documents = PyPDFLoader(str(file_path)).load()
        return self._enrich_metadata(file_path=file_path, documents=documents)


class EmailExtractor(Extractor):
    """Estrae il contenuto di un'email (.eml) tramite email.parser."""
    format = "email"

    def extract(self, file_path: Path) -> List[Document]:
        with open(file_path, 'rb') as file:
            parser = BytesParser(policy=policy.default)
            message = parser.parse(file)

        body = message.get_body()
        body_text = body.get_content() if body else ""

        headers = (
            f"Da: {message.get('From', '')}\n"
            f"A: {message.get('To', '')}\n"
            f"Oggetto: {message.get('Subject', '')}\n"
            f"Data: {message.get('Date', '')}\n\n"
        )

        document = Document(page_content=headers + body_text)
        return self._enrich_metadata(file_path=file_path, documents=[document])


class DocxExtractor(Extractor):
    """Estrae il contenuto di un documento Word tramite Docx2txtLoader di LangChain."""
    format = "docx"

    def extract(self, file_path: Path) -> List[Document]:
        documents = Docx2txtLoader(str(file_path)).load()
        return self._enrich_metadata(file_path=file_path, documents=documents)


class FormatManager:
    """Gestisce i formati di documento supportati (.docx, .pdf, .eml)."""
    def __init__(self):
        self.extractors: Dict[str, Extractor] = {
            ".eml": EmailExtractor(),
            ".pdf": PDFExtractor(),
            ".docx": DocxExtractor()
        }

    def extract_document(self, file_path: Path) -> Document:
        file_extension = file_path.suffix.lower()
        extractor = self.extractors.get(file_extension)

        if extractor is None:
            raise ValueError(f"Nessun estrattore disponibile per il formato: {file_path.suffix}")

        contents = extractor.extract(file_path=file_path)
        combined_text = "\n".join(p.page_content for p in contents).strip()
        metadata = {"file_name": file_path.name, "format": extractor.format}

        return Document(page_content=combined_text, metadata=metadata)


# --- CHUNKER ---
class DocumentChunker:
    """Sistema di suddivisione (chunking) dei documenti"""
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_content(self, content: Document) -> List[DocumentChunk]:
        """Divide il contenuto del documento in porzioni più piccole (chunks)."""
        chunks = []

        if not content.page_content.strip():
            return chunks

        text_chunks = self.text_splitter.split_text(content.page_content)
        for i, text_chunk in enumerate(text_chunks):
            chunk_id = f"{content.metadata.get('file_name')}_{content.metadata.get('format')}_{i}"
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=text_chunk,
                metadata={
                    **content.metadata,
                    'chunk_index': i,
                    'chunk_length': len(text_chunk)
                }
            )
            chunks.append(chunk)
        return chunks


# --- SISTEMA DI EMBEDDING ---
class EmbeddingManager:
    """
    Gestisce la generazione di vettori (embeddings) tramite modelli pre-addestrati.\
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode_text(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        """Genera gli embedding per i testi forniti in ingresso."""
        if not texts:
            return np.array([])

        return self.model.encode(
            texts,
            batch_size=batch_size
        )
class LLM(ABC):

    @property
    def chat_model(self):
        """Proprietà in sola lettura per accedere al modello Chat di LangChain."""
        return self._chat

# --- LLM Locale ---
class LocalLLM(LLM):
    """Espone un modello chat locale tramite HuggingFacePipeline e ChatHuggingFace."""

    def __init__(
        self,
        model_name: str = LOCAL_LLM_MODEL,
        device_map: str = DEVICE_MAP,
        max_new_tokens: int = MAX_TOKENS,
    ):
        print(f"Caricamento modello locale '{model_name}' (verrà scaricato al primo avvio)...")

        pipeline_hf = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            device_map=device_map,
            pipeline_kwargs={
                "max_new_tokens": max_new_tokens,
                "max_length": None,
                "do_sample": False,
                "return_full_text": False,
            },
        )
        self._chat = ChatHuggingFace(llm=pipeline_hf)
        print("Modello locale pronto.\n")


# --- LLM via API ---
class ApiLLM(LLM):
    """Espone un modello chat remoto tramite un endpoint compatibile OpenAI (Groq, OpenRouter, OpenAI...)."""

    def __init__(self, api_key):
        self._chat = ChatOpenAI(
            model=GROQ_MODEL,
            api_key=api_key,
            base_url=LLM_API_BASE_URL,
            reasoning_effort="low",
            temperature=0,
            max_retries=2,
            max_tokens=MAX_TOKENS
        )


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print(f"Chiave API mancante: imposta la variabile d'ambiente GROQ_API_KEY.")

    return ApiLLM(api_key) if api_key else LocalLLM()


# --- ARCHIVIO VETTORIALE CON CHROMADB ---
class VectorArchive:
    """Interfaccia verso il database vettoriale ChromaDB per la persistenza e il recupero."""
    def __init__(self, collection_name: str = COLLECTION_NAME, persist_dir: str = VECTOR_FOLDER):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Crea una nuova collezione per la sessione corrente
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self.total_chunks = 0

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        if chunks is None:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Il numero di chunk ({len(chunks)}) non corrisponde al numero di embedding ({len(embeddings)})"
            )

        batch_size = 100

        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            batch_embeddings = embeddings[batch_start:batch_end]

            ids = [c.chunk_id for c in batch_chunks]
            documents = [c.text for c in batch_chunks]
            metadatas = [c.metadata for c in batch_chunks]
            embeddings_list = [e.tolist() for e in batch_embeddings]

            try:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings_list
                )
                self.total_chunks += len(batch_chunks)
            except Exception as e:
                print(f"Errore durante l'aggiunta del batch: {e}")

            print(f"Totale chunk indicizzati nello store: {self.total_chunks}")

    def search(self, query_embedding: np.ndarray, file_name: Optional[str] = None, top_k: int = TOP_K_RETRIEVAL) -> List[Document]:
        """Esegue una ricerca nel vector store, filtrando opzionalmente per singolo file."""
        where_filter = {"file_name": file_name} if file_name else None

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )

        documents: List[Document] = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                documents.append(
                    Document(
                        page_content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i]
                    )
                )
        return documents

    def summarize_theme(self,llm, embedding_manager: EmbeddingManager, theme_query: str, top_k: int = TOP_K_RETRIEVAL) -> Dict[str, str]:
        """
        Recupera i chunk più rilevanti per un determinato tema su tutti i documenti e li sintetizza per ciascun file.
        """
        query_embedding = embedding_manager.encode_text([theme_query])[0]
        results = self.search(query_embedding, top_k=top_k)

        by_file: Dict[str, List[str]] = {}
        for doc in results:
            by_file.setdefault(doc.metadata.get("file_name", "doc"), []).append(doc.page_content)

        summaries = {}
        for fname, texts in by_file.items():
            combined_text = "\n\n".join(texts)
            response = llm.chat_model.invoke([
                HumanMessage(content=(
                    f"Riassumi in 3-5 frasi i punti chiave relativi al tema '{theme_query}' "
                    f"nel seguente testo: \n\n{combined_text}"
                ))
            ])
            summaries[fname] = response.content
        return summaries

    def build_thematic_report(self,llm, embedding_manager: EmbeddingManager) -> str:
        sections = []
        for theme_key, theme_query in THEMES.items():
            summaries = self.summarize_theme(llm, embedding_manager, theme_query)
            memo = self.build_final_memo(llm, theme_query, summaries)
            sections.append(f"# {theme_key.replace('_', ' ').title()}\n\n{memo}")
        return "\n\n---\n\n".join(sections)

    def build_final_memo(self,llm , query: str, summaries: Dict[str, str]) -> str:
        if not summaries:
            return "Nessun documento rilevante trovato per la query richiesta."

        per_doc_section = "\n\n".join(f"### {fname}\n{summary}" for fname, summary in summaries.items())

        messages = [
            SystemMessage(content=(
                "Sei un assistente specializzato in report operativi. Ti vengono forniti "
                "riassunti divisi per documento: NON modificarli. "
                "In coda ad essi, aggiungi due sezioni finali:\n"
                "## Sintesi tematica (in comune tra i documenti)\n"
                "## Action item operativi\n"
                "Se non ci sono elementi per una sezione, scrivi 'Nessuno rilevato'."
            )),
            HumanMessage(content=f"Argomento: {query}\n\nRiassunti per documento:\n\n{per_doc_section}"),
        ]
        response = llm.chat_model.invoke(messages)

        return f"{per_doc_section}\n\n{response.content}"


class DocumentAnalyzer:
    """Esegue l'estrazione dei metadati e la classificazione forzando l'LLM a rispondere in formato JSON."""

    _RECOVERY_QUERY = (
        "categoria del documento, importi, date, rischi, non conformità, "
        "criticità, decisioni e raccomandazioni"
    )

    def __init__(self,llm , archive: VectorArchive, embedding_manager: EmbeddingManager):
        self._archive = archive
        self._embedding_manager = embedding_manager
        self._parser = JsonOutputParser(pydantic_object=_LLMAnalysisSchema)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Sei un analista di compliance e finanza aziendale. "
                    "Rispondi ESCLUSIVAMENTE in formato JSON valido seguendo queste istruzioni:\n"
                    "{format_instructions}",
                ),
                (
                    "human",
                    "Analizza il seguente estratto del documento '{file_name}'.\n"
                    "Categorie ammesse per 'category': {categories}.\n\n"
                    "CONTENUTO DEL DOCUMENTO:\n\"\"\"\n{context}\n\"\"\"\n",
                ),
            ]
        )
        self._chain = prompt | llm.chat_model | self._parser

    def analyze(self, file_name: str, format: str) -> DocumentAnalysis:
        query_embedding = self._embedding_manager.encode_text([self._RECOVERY_QUERY])[0]
        chunks = self._archive.search(query_embedding, file_name=file_name)
        context = "\n\n---\n\n".join(c.page_content for c in chunks)

        try:
            data = self._chain.invoke(
                {
                    "file_name": file_name,
                    "categories": ", ".join(f'"{c}"' for c in CATEGORIES),
                    "context": context,
                    "format_instructions": self._parser.get_format_instructions(),
                }
            )
        except Exception:
            # Fallback se il modello LLM non rispetta la struttura JSON richiesta
            return DocumentAnalysis(
                file_name=file_name,
                format=format,
                category="Revisione manuale richiesta",
                number_of_chunks_analyzed=len(chunks),
            )

        category = data.get("category", "Non classificato")

        if category not in CATEGORIES:
            category = "Revisione manuale richiesta"

        try:
            confidence = round(float(data.get("category_confidence", 0.0)), 2)
        except (TypeError, ValueError):
            confidence = 0.0

        return DocumentAnalysis(
            file_name=file_name,
            format=format,
            category=category,
            category_confidence=confidence,
            summary=data.get("summary", ""),
            key_insights=data.get("key_insights") or [],
            detected_dates=data.get("recorded_dates") or [],
            detected_amounts=data.get("detected_amounts") or [],
            detected_entities=data.get("detected_entities") or [],
            critical_issues=data.get("critical_issues") or [],
            recommendation=data.get("recommendation", ""),
            number_of_chunks_analyzed=len(chunks),
        )

# PIPELINE
class DocumentAnalysisPipeline:
    """Orchestra tutti i componenti ed esegue l'analisi sequenziale sui documenti."""
    def __init__(self):
        self._format_manager = FormatManager()
        self._chunker = DocumentChunker()
        self._embedding_manager = EmbeddingManager()
        collection_name = f"{COLLECTION_NAME}_{uuid.uuid4().hex[:8]}"
        self._archive = VectorArchive(collection_name, VECTOR_FOLDER)
        self._llm = get_llm()
        self._analyzer = DocumentAnalyzer(self._llm,self._archive, self._embedding_manager)

    def run(self) -> List[DocumentAnalysis]:
        print("Recupero dei documenti dalla cartella di destinazione...")
        file_paths = [
            p for p in DATA_FOLDER.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        for path in file_paths:
            if not path.is_file():
                print(f"      - File non valido ignorato: {path.name}")
                continue

        print("\nEstrazione, chunking e indicizzazione in ChromaDB...")
        original_documents = []
        for file_path in file_paths:
            document = self._format_manager.extract_document(file_path)
            chunks = self._chunker.chunk_content(document)
            if not chunks:
                print(f"      - {file_path.name}: nessun chunk generato (documento vuoto), indicizzazione saltata")
                continue
            original_documents.append(document)
            text_chunks = [c.text for c in chunks]
            embeddings = self._embedding_manager.encode_text(text_chunks)
            self._archive.add_chunks(chunks, embeddings)
            print(f"      - {file_path.name}: {len(chunks)} chunk indicizzati")

        print("\nAnalisi dei documenti in corso (Retrieval Semantico + LLM Locale)...")
        results = []
        for document in original_documents:
            file_name = document.metadata["file_name"]
            format = document.metadata["format"]
            print(f"      - Analisi di: {file_name} ...")
            results.append(self._analyzer.analyze(file_name, format))

        print("\n--- Riepilogo ---")
        for res in results:
            print(f"  - {res.file_name:35s} -> {res.category:28s} (conf. {res.category_confidence})")

        print("\nGenerazione del report tematico trasversale...")
        thematic_report = self._archive.build_thematic_report(
            self._llm,
            self._embedding_manager
        )
        print("\n--- Report Tematico ---")
        print(thematic_report)

        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

        report_path = OUTPUT_FOLDER / "thematic_report.md"
        report_path.write_text(thematic_report, encoding="utf-8")
        print(f"\nReport tematico salvato in: {report_path}")

        json_path = OUTPUT_FOLDER / "analysis_results.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
        print(f"Risultati dell'analisi salvati in: {json_path}")

        return results


def main() -> None:
    DocumentAnalysisPipeline().run()


if __name__ == "__main__":
    main()