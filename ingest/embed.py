import numpy as np
import json
from sentence_transformers import SentenceTransformer
from functools import lru_cache

PATH_CHUNKS = 'data/processed/greenspan_chunks.json'   
PATH_EMBS = 'data/processed/greenspan_embs.npz'
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

with open(PATH_CHUNKS, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

def get_model():
    return SentenceTransformer(MODEL_NAME)

def main():
    with open(PATH_CHUNKS, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    textos = [c['text'] for c in chunks]
    ids = np.array([c['chunk_index'] for c in chunks])

    model = get_model()

    embs = np.array(model.encode(textos, normalize_embeddings=True))
    
    print('=' * 20, 'Check', '=' * 20)
    print(f"{'Max seq length':<28} {model.max_seq_length}")
    print(f"{'Dimensión del modelo':<28} {model.get_embedding_dimension()}")
    print(f"{'Dimensiones de los vectores':<28} {embs.shape}")
    print(f"{'Valores finitos':<28} {np.isfinite(embs).all()}")
    
    np.savez(PATH_EMBS, embeddings=embs, ids=ids, model_name=MODEL_NAME, normalized=True)

if __name__ == '__main__':
    main()
