# nlp_utils.py (V2 - Adım 2.4 Eklentisi)
import spacy
from spacy.tokens import Doc, Span # Span eklendi
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F # Cosine Similarity için eklendi
from rich.console import Console
from typing import List, Tuple # List, Tuple eklendi
import os

# os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

console = Console()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console.print(f"Using device: [bold {'green' if device.type == 'cuda' else 'yellow'}]{device}[/bold {'green' if device.type == 'cuda' else 'yellow'}]")
if device.type == 'cuda': console.print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

# --- spaCy ---
SPACY_MODEL_NAME = "en_core_web_lg"
_nlp = None

def load_spacy_model(model_name: str = SPACY_MODEL_NAME) -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        try:
            console.print(f"Loading spaCy model '{model_name}'...")
            _nlp = spacy.load(model_name)
            console.print(f"[green]spaCy model '{model_name}' loaded successfully (on CPU).[/green]")
        except OSError:
            console.print(f"[bold red]Error: spaCy model '{model_name}' not found.[/bold red]"); raise
    return _nlp

def process_text_spacy(text: str) -> spacy.tokens.Doc:
    spacy_nlp = load_spacy_model()
    if spacy_nlp: return spacy_nlp(text)
    raise RuntimeError("spaCy model could not be loaded or process failed.")

# --- Transformers (BERT) ---
BERT_MODEL_NAME = "bert-base-uncased"
_tokenizer = None
_bert_model = None

def load_bert(model_name: str = BERT_MODEL_NAME) -> tuple[AutoTokenizer, AutoModel]:
    global _tokenizer, _bert_model
    if _tokenizer is None:
        try:
            console.print(f"Loading BERT tokenizer for '{model_name}'...")
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            console.print(f"[green]BERT Tokenizer for '{model_name}' loaded successfully.[/green]")
        except Exception as e: console.print(f"[bold red]Error loading BERT tokenizer '{model_name}': {e}[/bold red]"); raise
    if _bert_model is None:
        try:
            console.print(f"Loading full BERT model '{model_name}'...")
            _bert_model = AutoModel.from_pretrained(model_name)
            _bert_model.to(device); _bert_model.eval()
            console.print(f"[green]Full BERT Model '{model_name}' loaded successfully to [bold]{device}[/bold].[/green]")
        except Exception as e: console.print(f"[bold red]Error loading full BERT model '{model_name}': {e}[/bold red]"); raise
    return _tokenizer, _bert_model

def get_bert_embeddings(text: str, max_length: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    try: tokenizer, model = load_bert()
    except Exception as e: raise RuntimeError(f"Failed to load BERT model or tokenizer: {e}")
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, padding=True, truncation=True)
    inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        try: outputs = model(**inputs_on_device)
        except Exception as e: console.print(f"[bold red]Error during BERT inference: {e}[/bold red]"); raise RuntimeError(f"BERT inference failed: {e}")
    last_hidden_state = outputs.last_hidden_state; pooler_output = outputs.pooler_output
    return last_hidden_state.cpu().detach(), pooler_output.cpu().detach()

def get_sentence_embedding(sentence_text: str, strategy: str = 'mean', max_length: int = 512) -> torch.Tensor:
    """ Tek bir cümlenin embedding'ini hesaplar. """
    # Not: Bu hala her cümle için ayrı ayrı model çağrısı yapar, verimli değil.
    # Daha iyisi: Tüm metni bir kerede işleyip, sonra cümle spanlarına göre ortalama almak.
    # Şimdilik bu basit versiyonla devam edelim.
    try:
        last_hidden_state, _ = get_bert_embeddings(sentence_text, max_length)
        if strategy == 'cls':
            return last_hidden_state[0, 0, :]
        elif strategy == 'mean':
            tokenizer, _ = load_bert()
            inputs = tokenizer(sentence_text, return_tensors="pt", max_length=max_length, padding=True, truncation=True)
            attention_mask = inputs['attention_mask'].cpu()
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            return mean_embeddings[0]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to get embedding for sentence '{sentence_text[:50]}...': {e}[/yellow]")
        # Hata durumunda sıfır vektörü döndürelim? Veya None? None daha iyi.
        return None


# --- YENİ FONKSİYON ---
def get_all_sentence_embeddings(doc: Doc, strategy: str = 'mean', max_length: int = 512) -> List[torch.Tensor | None]:
    """
    Bir spaCy Doc içerisindeki tüm cümlelerin BERT embedding'lerini hesaplar.
    (Mevcut haliyle her cümle için ayrı model çağrısı yapar - verimsiz)
    """
    console.print(f" -> Calculating BERT sentence embeddings for {len(list(doc.sents))} sentences (Strategy: {strategy})...", style="dim")
    embeddings = []
    for sent in doc.sents:
        embedding = get_sentence_embedding(sent.text, strategy=strategy, max_length=max_length)
        embeddings.append(embedding)
    console.print(f" -> Sentence embeddings calculation complete.", style="dim")
    return embeddings