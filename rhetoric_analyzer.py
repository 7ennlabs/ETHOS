# rhetoric_analyzer.py
import spacy
from spacy.tokens import Doc, Span, Token
from typing import List
import data_models # Düz yapı importu
from rich.console import Console
# VADER'ı import et
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

console = Console()
# VADER duygu analizcisini başlat
analyzer = SentimentIntensityAnalyzer()

# Basit Retorik İpuçları

# Güçlü Duygu Eşiği (VADER compound score için)
# -1 (çok negatif) ile +1 (çok pozitif) arasında.
# |score| > 0.5 genellikle belirgin bir duygu ifade eder.
STRONG_SENTIMENT_THRESHOLD = 0.5

# Superlative'ler (En üstünlük belirten sıfatlar/zarflar)
# spaCy'nin POS tag'lerini kullanabiliriz: 'JJS' (superlative adjective), 'RBS' (superlative adverb)
SUPERLATIVE_TAGS = {"JJS", "RBS"}

# Retorik Sorular (Basit: Soru işareti var mı?)
# Daha gelişmiş analiz gerekir ama başlangıç için bu yeterli.

def analyze_sentence_sentiment(sent: Span) -> List[data_models.Finding]:
    """VADER kullanarak cümlenin duygu skorunu analiz eder ve güçlü duyguları bulur."""
    findings = []
    # VADER'ın polarity_scores fonksiyonu bir dict döndürür: {'neg', 'neu', 'pos', 'compound'}
    vs = analyzer.polarity_scores(sent.text)
    compound_score = vs['compound']

    description = None
    severity = "Low" # Varsayılan

    if compound_score >= STRONG_SENTIMENT_THRESHOLD:
        description = f"Sentence potentially expresses strong positive sentiment (VADER score: {compound_score:.2f})."
        severity = "Medium"
    elif compound_score <= -STRONG_SENTIMENT_THRESHOLD:
        description = f"Sentence potentially expresses strong negative sentiment (VADER score: {compound_score:.2f})."
        severity = "Medium"

    if description:
        findings.append(data_models.Finding(
            finding_type="RhetoricalDevice",
            description=description,
            severity=severity,
            span_start=sent.start_char,
            span_end=sent.end_char,
            details={"device_type": "Strong Sentiment", "vader_score": vs}
        ))
    return findings

def detect_superlatives(sent: Span) -> List[data_models.Finding]:
    """Cümlede superlative (en üstünlük) ifadeleri arar."""
    findings = []
    superlative_words = []
    for token in sent:
        if token.tag_ in SUPERLATIVE_TAGS:
            superlative_words.append(token.text)

    if superlative_words:
        findings.append(data_models.Finding(
            finding_type="RhetoricalDevice",
            description=f"Use of superlative(s) detected: {', '.join(superlative_words)}.",
            severity="Low", # Tek başına zayıf bir gösterge
            span_start=sent.start_char,
            span_end=sent.end_char,
            details={"device_type": "Superlative", "words": superlative_words}
        ))
    return findings

def detect_rhetorical_questions(sent: Span) -> List[data_models.Finding]:
    """Cümlede soru işareti olup olmadığını kontrol eder (çok basit)."""
    findings = []
    if sent.text.strip().endswith("?"):
        findings.append(data_models.Finding(
            finding_type="RhetoricalDevice",
            description="Sentence ends with a question mark (potential rhetorical question).",
            severity="Low", # Bağlam olmadan bilmek zor
            span_start=sent.start_char,
            span_end=sent.end_char,
            details={"device_type": "Potential Question"}
        ))
    return findings

def simple_rhetoric_analyzer(doc: Doc) -> List[data_models.Finding]:
    """
    Metindeki cümleleri basit kurallarla analiz ederek potansiyel retorik araçları bulur.
    """
    all_findings = []
    for sent in doc.sents:
        # Her cümle için retorik kontrollerini çalıştır
        all_findings.extend(analyze_sentence_sentiment(sent))
        all_findings.extend(detect_superlatives(sent))
        all_findings.extend(detect_rhetorical_questions(sent))
        # Buraya diğer retorik fonksiyon çağrıları eklenebilir

    console.print(f" -> Simple Rhetoric Analyzer found {len(all_findings)} potential rhetorical indicators.", style="dim")
    return all_findings