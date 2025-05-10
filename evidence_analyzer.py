# evidence_analyzer.py (V2 - Adım 2.4 Revizyonu: Basitleştirilmiş : GEREKLİYDİ ZAYN BAŞARAMADI :))
import spacy
from spacy.tokens import Doc, Span
from typing import List, Tuple
import data_models
# Artık nlp_utils, torch, F'e gerek yok bu basit versiyonda
from rich.console import Console
import re

console = Console()

# Sabitler (Aynı)
CITATION_PHRASES = {
    "according to", "study shows", "research indicates", "data suggests",
    "experts say", "report finds", "source:", "evidence shows", "demonstrates that",
    "reported by", "stated by", "cited in"
}
URL_REGEX = r"(?:https?://|www\.)[^\s/$.?#].[^\s]*"

# Yardımcı Fonksiyon (Aynı)
def has_potential_evidence_indicator(sent: Span) -> Tuple[bool, str, str]:
    sent_text = sent.text; sent_text_lower = sent.text.lower()
    if not sent_text.strip(): return False, "", ""
    urls = re.findall(URL_REGEX, sent_text)
    if urls: return True, "URL", urls[0]
    if re.search(r"\b\d{3,}\b", sent_text) or '%' in sent_text or re.search(r"\b\d+(?:\.\d+)?\b", sent_text):
        match = re.search(r"\b\d+(?:\.\d+)?%?\b", sent_text)
        trigger_text = match.group(0) if match else "Number/Percentage"
        return True, "Numerical Data", trigger_text
    for phrase in CITATION_PHRASES:
        if f" {phrase} " in f" {sent_text_lower} " or sent_text_lower.startswith(f"{phrase} "):
            return True, "Citation Phrase", phrase
    return False, "", ""

# Ana Analiz Fonksiyonu (Basitleştirilmiş - Sadece aynı cümleyi kontrol eder)
def simplified_evidence_analyzer(
    doc: Doc,
    argument_components: List[data_models.ArgumentComponent]
    # sentence_embeddings parametresi kaldırıldı
) -> List[data_models.Finding]:
    """
    Tespit edilen iddiaları (Claim) inceler ve SADECE kendi cümlelerinde
    basit kanıt göstergeleri olup olmadığını kontrol eder (V1 Tarzı Basit).
    """
    findings = []
    claims_data = [(idx, comp) for idx, comp in enumerate(argument_components) if comp.component_type == "Claim"]
    sentences = list(doc.sents)
    num_sentences = len(sentences)

    if not claims_data:
        console.print(" -> No claims found to analyze for evidence.", style="dim"); return findings

    console.print(f" -> Analyzing {len(claims_data)} claims for evidence indicators (Simplified: Same sentence only)...", style="dim")

    for claim_comp_idx, claim in claims_data:
        claim_sentence_idx = claim.sentence_index
        claim_text_snippet = claim.text[:100] + "..."

        if not (0 <= claim_sentence_idx < num_sentences):
            console.print(f"[yellow]Warn: Invalid sentence index {claim_sentence_idx} for claim comp_idx {claim_comp_idx}, skipping.[/yellow]"); continue

        claim_sentence_span = sentences[claim_sentence_idx]

        # Sadece iddianın kendi cümlesini kontrol et
        has_indicator, indicator_type, indicator_text = has_potential_evidence_indicator(claim_sentence_span)

        if has_indicator:
            # Gösterge varsa EvidenceIndicator ekle
            findings.append(data_models.Finding(
                finding_type="EvidenceIndicator",
                description=f"Potential evidence indicator ('{indicator_type}') found in the same sentence as the claim.",
                severity="Info",
                span_start=claim_sentence_span.start_char,
                span_end=claim_sentence_span.end_char,
                details={
                    "indicator_type": indicator_type, "indicator_trigger": indicator_text,
                    "location": "same_sentence", "linked_claim_index": claim_comp_idx,
                    "claim_text": claim_text_snippet
                }
            ))
        else:
            # Gösterge yoksa EvidenceStatus ekle
            findings.append(data_models.Finding(
                finding_type="EvidenceStatus",
                description="Claim lacks explicit evidence indicator in the same sentence.", # Açıklama basitleşti
                severity="Medium",
                span_start=claim_sentence_span.start_char, # İddia cümlesinin span'ı
                span_end=claim_sentence_span.end_char,
                details={"claim_text": claim_text_snippet}
            ))

    console.print(f" -> Simplified Evidence Analyzer generated {len(findings)} findings.", style="dim")
    return findings