# data_models.py
from typing import List, Optional, Any
from pydantic import BaseModel, Field

class Finding(BaseModel):
    """Analiz sırasında bulunan genel bir bulguyu temsil eder."""
    finding_type: str = Field(..., description="Bulgunun türü (örn. 'Fallacy', 'RhetoricalDevice', 'EvidenceStatus')")
    description: str = Field(..., description="Bulgunun kısa açıklaması")
    severity: Optional[str] = Field(None, description="Bulgunun ciddiyeti (örn. 'Low', 'Medium', 'High')")
    span_start: Optional[int] = Field(None, description="Bulgunun metindeki başlangıç karakter indeksi")
    span_end: Optional[int] = Field(None, description="Bulgunun metindeki bitiş karakter indeksi")
    details: Optional[dict[str, Any]] = Field(None, description="Bulguya özel ek detaylar")

class ArgumentComponent(BaseModel):
    """Tespit edilen bir argüman bileşenini temsil eder (İddia, Gerekçe vb.)."""
    component_type: str = Field(..., description="Bileşenin türü (örn. 'Claim', 'Premise')")
    text: str = Field(..., description="Bileşenin tam metni")
    sentence_index: int = Field(..., description="Bileşenin bulunduğu cümlenin indeksi")
    span_start: int = Field(..., description="Bileşenin cümle içindeki başlangıç karakter indeksi")
    span_end: int = Field(..., description="Bileşenin cümle içindeki bitiş karakter indeksi")
    confidence: Optional[float] = Field(None, description="Tespitin güven skoru (0.0 - 1.0)")

class SentenceInfo(BaseModel):
    """Tek bir cümle hakkındaki bilgileri içerir."""
    text: str
    start_char: int
    end_char: int
    tokens: List[str] # Şimdilik basitçe token metinleri

class AnalyzedText(BaseModel):
    """Tüm analiz sürecinin sonucunu içeren ana model."""
    original_text: str
    processed_text: Optional[str] = None
    language: str = "en"
    sentences: List[SentenceInfo] = []
    findings: List[Finding] = [] # Tüm bulgular burada toplanacak
    argument_components: List[ArgumentComponent] = []
    analysis_summary: Optional[dict[str, str]] = Field(None, description="Analizin özet değerlendirmesi") # YENİ ALAN

    class Config:
        extra = 'forbid'