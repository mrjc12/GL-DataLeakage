import os
import sys
import argparse
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm  


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def load_text_data(dataset, text_type, root_dir='processed_data'):
    print(f"  Loading data for {text_type}...", end=" ", flush=True)
    if text_type == 'NS':
        fmt = 'sbert'
        if dataset == 'arxiv':
            path = os.path.join(root_dir, f'{dataset}_fixed_{fmt}.pt')
        else:
            path = os.path.join(root_dir, f'{dataset}_fixed_{fmt}.pt')
            if not os.path.exists(path):
                path = os.path.join(root_dir, f'{dataset}_random_{fmt}.pt')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        data = torch.load(path, weights_only=False)
        print("Done.")
        return data['raw_texts']
    else:
        # For E, R, M, NE, etc.
        fname = text_type.split('_')[-1]

        path = os.path.join(root_dir, f'{dataset}_{fname}.pt')
        if not os.path.exists(path):
            path = os.path.join(root_dir, f'{dataset}_{text_type}.pt')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Text feature file not found: {path}")

        print("Done.")
        return torch.load(path, weights_only=False)


class TfidfEmbedder:
    def __init__(self, dataset):
        if dataset == 'cora':
            self.dim = 1433
        elif dataset == 'pubmed':
            self.dim = 500
        else:
            self.dim = 1000

    def encode(self, texts, device):
        print(f"  Fitting TF-IDF (dim={self.dim})...", end=" ", flush=True)
        vectorizer = TfidfVectorizer(max_features=self.dim, stop_words='english')
        X = vectorizer.fit_transform(texts)
        print("Done.")
        return X.toarray().astype(np.float16)


class QwenEmbedder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        print(f"Loading Model {model_name}...", end=" ", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left',trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name,trust_remote_code=True)
        self.model.eval()
        print("Done.")

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]

    def encode(self, texts, device="cuda", batch_size=16):
        self.model.to(device)
        all_embs = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Inferencing", unit="batch", ncols=100):
            batch = texts[i:i + batch_size]
            
            inputs = self.tokenizer(batch,return_tensors="pt",padding=True,truncation=True,max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self.last_token_pool(outputs.last_hidden_state,inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embs.append(embeddings.cpu().numpy().astype(np.float16))
        
        return np.concatenate(all_embs, axis=0)


class E5Embedder:
    def __init__(self, model_name="intfloat/multilingual-e5-large"):
        print(f"  Loading Model {model_name}...", end=" ", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        print("Done.")

    def encode(self, texts, device, batch_size=32):
        self.model.to(device)
        all_embs = []
        texts = ["query: " + x for x in texts]
        iterator = range(0, len(texts), batch_size)
        for i in tqdm(iterator, desc="  Inferencing", unit="batch", ncols=100):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                last_hidden = outputs.last_hidden_state
                last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
                embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            all_embs.append(embeddings.cpu().numpy().astype(np.float16))
        return np.concatenate(all_embs, axis=0)

class MiniLMEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"  Loading Model {model_name}...", end=" ", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        print("Done.")

    def encode(self, texts, device, batch_size=32):
        self.model.to(device)
        all_embs = []
        iterator = range(0, len(texts), batch_size)
        for i in tqdm(iterator, desc="  Inferencing", unit="batch", ncols=100):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                last_hidden = outputs.last_hidden_state
                last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
                embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            all_embs.append(embeddings.cpu().numpy().astype(np.float16))
        return np.concatenate(all_embs, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--text_types', nargs='+', required=True)
    parser.add_argument('--model', type=str, required=True, choices=['tfidf', 'qwen', 'e5', 'minilm'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    print(f"=== Generating {args.model} embeddings for {args.dataset} (Seed: {args.seed}) ===")

    # Setup Embedder
    if args.model == 'tfidf':
        embedder = TfidfEmbedder(args.dataset)
        model_name_str = "tfidf"
    elif args.model == 'e5':
        embedder = E5Embedder("intfloat/multilingual-e5-large")
        model_name_str = "intfloat/multilingual-e5-large"
    elif args.model == 'minilm':
        embedder = MiniLMEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        model_name_str = "sentence-transformers/all-MiniLM-L6-v2"
    else:
        embedder = QwenEmbedder("Qwen/Qwen3-Embedding-0.6B")
        model_name_str = "Qwen/Qwen3-Embedding-0.6B"

    for text_type in args.text_types:
        try:
            print(f"\n[Task: {text_type}]")
            texts = load_text_data(args.dataset, text_type)

            embeddings = embedder.encode(texts, device)
            num_nodes, dim = embeddings.shape

            # Save Path
            out_path = f"prt_lm/{args.dataset}/{model_name_str}_{text_type}-seed{args.seed}.emb"
            ensure_dir(out_path)

            # Write memmap
            print(f"  Saving to disk...", end=" ", flush=True)
            fp = np.memmap(out_path, dtype=np.float16, mode='w+', shape=(num_nodes, dim))
            fp[:] = embeddings[:]
            fp.flush()
            print(f"Done.\n  -> Saved to: {out_path} (Shape: {num_nodes}x{dim})")

        except Exception as e:
            print(f"\n  [ERROR] Processing {text_type}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()