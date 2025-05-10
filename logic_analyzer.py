# logic_analyzer.py (V2 - Adım 2.4 Düzeltmesi - Fonksiyon Adı Hatası Giderildi)
import spacy
from spacy.tokens import Doc, Span
from typing import List, Dict, Tuple
import data_models # Düz yapı importu
import nlp_utils   # Embedding fonksiyonları için
import torch
import torch.nn as nn
from rich.console import Console

console = Console()

# --- Basit Kural Tabanlı Safsata İpuçları (Aynı) ---
POPULUM_INDICATORS = {
    "everybody knows", "everyone knows", "everyone agrees",
    "it is common sense", "most people think", "the majority believes"
}
HASTY_GENERALIZATION_KEYWORDS = {"always", "never", "all", "none", "every", "everyone", "nobody"}
def check_false_dichotomy(sent_text_lower: str) -> bool:
    has_either = " either " in f" {sent_text_lower} " or sent_text_lower.startswith("either ")
    has_or = " or " in f" {sent_text_lower} "
    return has_either and has_or

# Kural Tabanlı Tespit Fonksiyonları (Aynı)
def detect_ad_populum(sent: Span) -> List[data_models.Finding]:
    findings = []; sent_text_lower = sent.text.lower()
    for indicator in POPULUM_INDICATORS:
        if indicator in sent_text_lower:
            findings.append(data_models.Finding(finding_type="Fallacy", description="Potential 'Appeal to Popularity' (Ad Populum / Bandwagon) detected by rule.", severity="Medium", span_start=sent.start_char, span_end=sent.end_char, details={"fallacy_type": "Ad Populum (Rule)", "trigger": indicator})); break
    return findings

def detect_hasty_generalization(sent: Span) -> List[data_models.Finding]:
    findings = []
    for token in sent:
        if token.text.lower() in HASTY_GENERALIZATION_KEYWORDS:
            findings.append(data_models.Finding(finding_type="Fallacy", description="Potential 'Hasty Generalization' detected by keyword rule (needs context!).", severity="Low", span_start=sent.start_char, span_end=sent.end_char, details={"fallacy_type": "Hasty Generalization (Rule)", "trigger": token.text})); break
    return findings

def detect_false_dichotomy(sent: Span) -> List[data_models.Finding]:
    findings = []; sent_text_lower = sent.text.lower()
    if check_false_dichotomy(sent_text_lower):
        findings.append(data_models.Finding(finding_type="Fallacy", description="Potential 'False Dichotomy' (Either/Or Fallacy) detected by rule.", severity="Medium", span_start=sent.start_char, span_end=sent.end_char, details={"fallacy_type": "False Dichotomy (Rule)", "trigger": "either...or pattern"}))
    return findings

# --- ML Tabanlı Safsata Tespiti (Placeholder - Fonksiyon adı düzeltildi) ---
FALLACY_CLASSES = ["Ad Hominem", "Hasty Generalization", "Appeal to Popularity", "No Fallacy"]
BERT_HIDDEN_SIZE = 768; NUM_FALLACY_CLASSES = len(FALLACY_CLASSES)

class FallacyClassifierPlaceholder(nn.Module):
    def __init__(self, input_size=BERT_HIDDEN_SIZE, num_classes=NUM_FALLACY_CLASSES):
        super().__init__(); self.linear = nn.Linear(input_size, num_classes)
        # Bu mesajın sadece bir kere görünmesi için kontrol eklenebilir ama şimdilik kalsın
        console.print("[yellow]Placeholder Fallacy Classifier initialized (UNTRAINED). Results will NOT be accurate.[/yellow]", style="dim")
    def forward(self, sentence_embedding):
        if sentence_embedding.dim() == 1: sentence_embedding = sentence_embedding.unsqueeze(0)
        logits = self.linear(sentence_embedding); return logits

try:
    placeholder_classifier = FallacyClassifierPlaceholder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    placeholder_classifier.to(device); placeholder_classifier.eval()
    _ml_classifier_loaded = True
except Exception as e:
    console.print(f"[bold red]Error initializing ML Fallacy Classifier: {e}. ML detection disabled.[/bold red]")
    _ml_classifier_loaded = False


def ml_fallacy_detection_for_sentence(sentence_text: str, sent_span: Span) -> List[data_models.Finding]: # Span eklendi
    """
    Tek bir cümle için BERT embedding'i alır ve EĞİTİLMEMİŞ sınıflandırıcı ile
    safsata tahmini yapar (PLACEHOLDER - DOĞRU SONUÇ VERMEZ).
    Span bilgisi eklendi.
    """
    findings = []
    if not _ml_classifier_loaded: return findings # Sınıflandırıcı yüklenemediyse boş dön

    try:
        # DÜZELTME: Doğru fonksiyon adını kullan: get_sentence_embedding (tekil)
        sentence_embedding = nlp_utils.get_sentence_embedding(sentence_text, strategy='mean')
        if sentence_embedding is None: # Embedding alınamadıysa
             return findings

        sentence_embedding = sentence_embedding.to(device)

        with torch.no_grad(): logits = placeholder_classifier(sentence_embedding)

        probabilities = torch.softmax(logits.squeeze(), dim=0)
        predicted_prob, predicted_idx = torch.max(probabilities, dim=0)
        predicted_class = FALLACY_CLASSES[predicted_idx.item()]
        predicted_prob = predicted_prob.item()

        # Sadece "No Fallacy" olmayanları ekle (güvenilmez skorla)
        if predicted_class != "No Fallacy":
            findings.append(data_models.Finding(
                finding_type="Fallacy",
                description=f"Potential '{predicted_class}' detected by ML Placeholder (Score: {predicted_prob:.2f} - UNRELIABLE).",
                severity="Low",
                span_start=sent_span.start_char, # Doğru span bilgisi kullanıldı
                span_end=sent_span.end_char,     # Doğru span bilgisi kullanıldı
                details={"fallacy_type": f"{predicted_class} (ML Placeholder)", "confidence": predicted_prob}
            ))
    except Exception as e:
        console.print(f"[yellow]Warning: ML Fallacy prediction failed for sentence: {e}[/yellow]", style="dim")
    return findings


# --- Geliştirilmiş Ana Analiz Fonksiyonu (Düzeltilmiş Hali) ---
def enhanced_fallacy_analyzer(doc: Doc) -> List[data_models.Finding]:
    """
    Metindeki cümleleri hem basit kurallarla hem de ML Placeholder ile
    analiz ederek potansiyel safsataları bulur (V2 Seviyesi).
    """
    all_findings = []
    console.print(" -> Running Rule-Based Fallacy Checks...", style="dim")
    sentences = list(doc.sents)
    for sent in sentences:
        all_findings.extend(detect_ad_populum(sent))
        all_findings.extend(detect_hasty_generalization(sent))
        all_findings.extend(detect_false_dichotomy(sent))

    if _ml_classifier_loaded: # ML sınıflandırıcı yüklendiyse çalıştır
        console.print(f" -> Running ML Placeholder Fallacy Checks ({len(sentences)} sentences)...", style="dim")
        for sent in sentences:
            # ml_fallacy_detection_for_sentence'a artık span'ı da gönderiyoruz
            all_findings.extend(ml_fallacy_detection_for_sentence(sent.text, sent))
    else:
         console.print(" -> Skipping ML Placeholder Fallacy Checks (Initialization failed).", style="dim")

    # TODO: Bulguları birleştirme / önceliklendirme
    console.print(f" -> Enhanced Fallacy Analyzer found {len(all_findings)} potential indicators (Rules + ML Placeholder).", style="dim")
    return all_findings