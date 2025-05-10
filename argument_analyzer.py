# argument_analyzer.py (V2 - Adım 2.2 Düzeltilmiş Hali - Tekrar)
import spacy
from spacy.tokens import Doc, Span, Token
from typing import List, Tuple, Dict, Set
import data_models # Düz yapı importu
from rich.console import Console

console = Console()

# --- Göstergeler ---
CLAIM_INDICATORS_VERBS = {"believe", "think", "argue", "claim", "suggest", "conclude", "state", "assert", "maintain", "propose", "insist", "contend", "show", "demonstrate", "indicate"}
CLAIM_INDICATORS_PHRASES = {"in my opinion", "it seems that", "the main point is", "it is clear that", "clearly", "the conclusion is", "we must", "it is necessary", "it is evident that"}
MODAL_VERBS_STRONG = {"should", "must", "ought", "has to", "needs to"}
PREMISE_INDICATORS = {"because", "since", "as", "for", "given that", "due to", "owing to", "assuming that", "the reason is", "if"}

# --- Dependency Parsing Yardımcı Fonksiyonları ---
def find_root_verb(sent: Span) -> Token | None:
    for token in sent:
        if token.dep_ == "ROOT" and token.pos_ == "VERB": return token
    for token in sent:
         if token.dep_ == "ROOT" and token.pos_ == "AUX": return token
    return None

def has_subject(verb: Token) -> bool:
    return any(child.dep_ in {"nsubj", "nsubjpass"} for child in verb.children)

def has_complement_or_object(verb: Token) -> bool:
    return any(child.dep_ in {"dobj", "attr", "acomp", "ccomp", "xcomp", "pobj"} for child in verb.children)

def get_clause_starting_with(indicator: str, sent: Span) -> Span | None:
     indicator_token = None
     for token in sent:
          if token.text.lower() == indicator and token.dep_ in {"mark", "prep", "agent"}:
               indicator_token = token; break
     if indicator_token:
          start_char = indicator_token.idx; end_char = sent.end_char
          clause_span = sent.doc.char_span(start_char, end_char)
          return clause_span if clause_span else sent
     return None

# --- Gelişmiş Analiz Fonksiyonu (Düzeltilmiş Hali) ---
def enhanced_component_analyzer(doc: Doc) -> List[data_models.ArgumentComponent]:
    """
    spaCy Doc nesnesini analiz ederek dependency parsing ve göstergelerle
    potansiyel iddia ve gerekçeleri tespit eder (V2 - Düzeltilmiş).
    """
    components = []
    sentences = list(doc.sents)

    for i, sent in enumerate(sentences):
        sent_text_lower = sent.text.lower()
        potential_components_in_sentence: List[Tuple[str, float, Span]] = []
        premise_found_in_this_sentence = False

        # --- 1. Gerekçe Kontrolü ---
        for indicator in PREMISE_INDICATORS:
            if f" {indicator} " in f" {sent_text_lower} " or sent_text_lower.startswith(f"{indicator} "):
                span_to_use = sent # Şimdilik tüm cümleyi al
                potential_components_in_sentence.append(("Premise", 0.65, span_to_use))
                premise_found_in_this_sentence = True
                break

        if premise_found_in_this_sentence:
            comp_type, confidence, span = potential_components_in_sentence[0]
            components.append(data_models.ArgumentComponent(
                 component_type=comp_type, text=span.text, sentence_index=i,
                 span_start=span.start_char, span_end=span.end_char, confidence=confidence
            ))
            continue # Diğer cümleye geç

        # --- Gerekçe bulunamadıysa İddia Kontrolleri ---
        claim_indicator_confidence = 0.0
        has_claim_verb = any(token.lemma_ in CLAIM_INDICATORS_VERBS and token.pos_ == "VERB" for token in sent)
        has_modal = any(token.lemma_ in MODAL_VERBS_STRONG and token.pos_ == "AUX" for token in sent)
        has_claim_phrase = any(phrase in sent_text_lower for phrase in CLAIM_INDICATORS_PHRASES)
        if has_claim_verb: claim_indicator_confidence = max(claim_indicator_confidence, 0.5)
        if has_modal: claim_indicator_confidence = max(claim_indicator_confidence, 0.6)
        if has_claim_phrase: claim_indicator_confidence = max(claim_indicator_confidence, 0.7)
        if claim_indicator_confidence > 0:
             potential_components_in_sentence.append(("Claim", claim_indicator_confidence, sent))

        root_verb = find_root_verb(sent)
        if root_verb:
            if has_subject(root_verb) and has_complement_or_object(root_verb):
                 if not any(c[0] == "Claim" for c in potential_components_in_sentence):
                      potential_components_in_sentence.append(("Claim", 0.4, sent))

        best_component = None; max_confidence = 0.0
        for comp_type, confidence, span in potential_components_in_sentence:
             if confidence > max_confidence:
                  max_confidence = confidence; best_component = (comp_type, span)

        if best_component:
            comp_type, span = best_component
            components.append(data_models.ArgumentComponent(
                 component_type=comp_type, text=span.text, sentence_index=i,
                 span_start=span.start_char, span_end=span.end_char, confidence=max_confidence
            ))

    console.print(f" -> Enhanced Analyzer (Corrected Version) found {len(components)} potential components.", style="dim")
    return components