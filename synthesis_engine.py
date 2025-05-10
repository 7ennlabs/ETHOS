# synthesis_engine.py (V2 - Adım 2.4 Revizyonu: Kanıt Değerlendirmesi Kaldırıldı)
from typing import List, Dict, Tuple
import data_models
from rich.console import Console

console = Console()

# Eşik Değerleri (Kanıt kaldırıldı)
FALLACY_THRESHOLD_LOW = 1
FALLACY_THRESHOLD_MEDIUM = 0
# EVIDENCE_LACK_THRESHOLD_WEAK = 0.5 # Kaldırıldı
# EVIDENCE_LACK_THRESHOLD_MODERATE = 0.1 # Kaldırıldı
RHETORIC_THRESHOLD_HIGH = 5
RHETORIC_THRESHOLD_MEDIUM = 2

def generate_summary_ratings(
    components: List[data_models.ArgumentComponent],
    findings: List[data_models.Finding]
) -> Dict[str, str]:
    """
    Bulunan bileşenlere ve bulgulara göre basit özet değerlendirmeler üretir.
    (Kanıt değerlendirmesi kaldırıldı).
    """
    summary = {}

    # --- Mantıksal Sağlamlık ---
    fallacies = [f for f in findings if f.finding_type == "Fallacy"]
    num_fallacies = len(fallacies)
    # TODO: ML modelinden gelen güven skorları da değerlendirmeye katılabilir
    if num_fallacies >= FALLACY_THRESHOLD_LOW:
        summary["Logical Soundness"] = "Low (Potential fallacies detected)"
    elif num_fallacies == FALLACY_THRESHOLD_MEDIUM:
        summary["Logical Soundness"] = "Medium (No obvious fallacies detected by current rules/model)"
    else: # Bu durum ML placeholder'da zor ama teorik olarak
         summary["Logical Soundness"] = "High (Potentially sound)"


    # --- Kanıtsal Dayanak ---
    # BU BÖLÜM KALDIRILDI - Güvenilir analiz yapılamadığı için yorumda bırakıldı veya silindi.
    # claims = [c for c in components if c.component_type == "Claim"]
    # num_claims = len(claims)
    # claims_lacking_evidence = sum(1 for f in findings if f.finding_type == "EvidenceStatus")
    # if num_claims == 0:
    #     summary["Evidential Basis"] = "N/A (No claims identified)"
    # elif claims_lacking_evidence == num_claims:
    #      summary["Evidential Basis"] = "Weak (All identified claims lack evidence indicators)"
    # # ... (diğer eski kontroller) ...
    summary["Evidential Basis"] = "Not Evaluated" # Geçici olarak devre dışı bırakıldı


    # --- Retorik Bütünlük ---
    rhetorical_findings = [f for f in findings if f.finding_type == "RhetoricalDevice"]
    num_rhetoric = len(rhetorical_findings)
    if num_rhetoric >= RHETORIC_THRESHOLD_HIGH:
        summary["Rhetorical Clarity"] = "Questionable (High use of rhetorical devices detected)"
    elif num_rhetoric >= RHETORIC_THRESHOLD_MEDIUM:
        summary["Rhetorical Clarity"] = "Mixed (Some rhetorical devices detected)"
    else:
        summary["Rhetorical Clarity"] = "Appears Clear (Few rhetorical devices detected)"


    console.print(f" -> Synthesis engine generated summary ratings (Evidence analysis excluded).", style="dim")
    return summary