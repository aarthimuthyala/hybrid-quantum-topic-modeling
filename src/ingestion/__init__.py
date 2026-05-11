"""src.ingestion — Data Ingestion Layer (Team T-1)."""
from src.ingestion.corpus_loader import load_corpus, load_corpus_from_file, load_corpus_from_url
from src.ingestion.text_preprocessor import TextPreprocessor, CleanDocument, preprocess_corpus, PreprocessingConfig

__all__ = [
    "load_corpus",
    "load_corpus_from_file",
    "load_corpus_from_url",
    "TextPreprocessor",
    "CleanDocument",
    "PreprocessingConfig",
    "preprocess_corpus",
]