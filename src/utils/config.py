#import os
from pathlib import Path

# Questa riga trova la cartella 'rag-edgar-system' partendo dalla posizione di questo file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Definiamo i percorsi principali basandoci sulla Root
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"

# Crea le cartelle se non esistono (così non avrai mai errori di 'Folder not found')
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHUNKS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Esporta i percorsi come stringhe per comodità
RAW_PATH = str(RAW_DATA_DIR)
CHUNKS_PATH = str(CHUNKS_DIR)