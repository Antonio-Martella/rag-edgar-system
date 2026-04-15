# src/ingestion/__init__.py

from .downloader import EdgarDownloader
from .parser import extract_pure, linearize_sec_tables, clean_sec_text
from .chunking import create_chunks
from .pipeline import run_ingestion_pipeline

# Definiamo cosa è "pubblico" e accessibile dall'esterno
__all__ = [
    "EdgarDownloader",
    "extract_pure",
    "linearize_sec_tables",
    "clean_sec_text",
    "create_chunks",
    "run_ingestion_pipeline"
]