# cli.py (V2/V3 - Adım 3.1: Anlamsal Görselleştirme Entegrasyonu)
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.padding import Padding
import data_models
import nlp_utils
import argument_analyzer
import logic_analyzer
# import evidence_analyzer # Kaldırıldı
import rhetoric_analyzer
import synthesis_engine
import argument_visualizer # Güncellenmiş versiyonu çağıracak
import networkx as nx
from typing import Optional, List
import sys
import textwrap

console = Console()
sentences = []

# --- Yardımcı Raporlama Fonksiyonları (Aynı) ---
def format_sentence_with_highlights(
    sentence_idx: int, all_sentences: List[data_models.SentenceInfo],
    findings: List[data_models.Finding], components: List[data_models.ArgumentComponent]
) -> Text:
    # ... (Fonksiyon içeriği öncekiyle aynı)...
    if sentence_idx < 0 or sentence_idx >= len(all_sentences): return Text("(Error: Invalid sentence index)")
    sentence = all_sentences[sentence_idx]; text = Text(sentence.text)
    for comp in components:
        if comp.sentence_index == sentence_idx:
             style = "bold magenta" if comp.component_type == "Claim" else "magenta"
             start = comp.span_start - sentence.start_char; end = comp.span_end - sentence.start_char
             if 0 <= start < end <= len(sentence.text):
                 try: text.stylize(style, start, end)
                 except Exception as e: console.print(f"[dim yellow]Warn: Styling component ({start}-{end}) in sent {sentence_idx}: {e}[/dim]")
    for finding in findings:
         if finding.span_start == sentence.start_char and finding.span_end == sentence.end_char:
              prefix = "[F] " if finding.finding_type == "Fallacy" else \
                       "[R] " if finding.finding_type == "RhetoricalDevice" else "[?] "
              style = "bold red" if prefix=="[F] " else "bold yellow" if prefix=="[R] " else ""
              text.insert(0, prefix, style=style)
    return text

# Ana CLI Fonksiyonu
def main(
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Text to analyze directly."),
    file_path: Optional[str] = typer.Option(None, "--file", "-f", help="Path to a text file to analyze."),
    max_findings_display: int = typer.Option(5, "--max-findings", "-m", help="Max number of each finding type to display in detail.")
):
    """
    ETHOS: The AI Arbiter of Rational Discourse (CLI) - v2.2
    Semantic Argument Linking & AI Fallacy Integration.
    """
    console.print(Panel("[bold cyan]ETHOS Analysis Engine v2.2 Starting...[/bold cyan]", expand=False, border_style="cyan"))
    # --- Girdi Kontrolü ve Yükleme ---
    if text and file_path: console.print("[bold red]Error: Use --text OR --file, not both.[/bold red]"); raise typer.Exit(code=1)
    if not text and not file_path: console.print("[bold red]Error: Use --text '...' OR --file '...'[/bold red]"); raise typer.Exit(code=1)
    text_to_analyze = ""; input_source_msg = ""
    if file_path:
        try:
            with open(file_path, 'r', encoding='utf-8') as f: text_to_analyze = f.read()
            input_source_msg = f"File: [yellow]'{file_path}'[/yellow]"
        except Exception as e: console.print(f"[bold red]Error reading file '{file_path}': {e}[/bold red]"); raise typer.Exit(code=1)
    elif text:
        text_to_analyze = text; input_source_msg = f"Input Text ({len(text_to_analyze)} chars)"
    if not text_to_analyze.strip(): console.print("[bold red]Error: Input text is empty.[/bold red]"); raise typer.Exit(code=1)
    console.print(Padding(f"Analyzing: {input_source_msg}", (0, 1)))

    # --- Analiz Adımları ---
    console.print("\n[bold blue]--- Initializing Analyzers & Embeddings ---[/bold blue]")
    try:
        nlp_utils.load_spacy_model(); nlp_utils.load_bert()
        spacy_doc = nlp_utils.process_text_spacy(text_to_analyze)
        # Tüm cümle embeddinglerini başta hesapla
        sentence_embeddings = nlp_utils.get_all_sentence_embeddings(spacy_doc)
    except Exception as e: console.print(f"[bold red]Error during model loading/processing/embedding: {e}[/bold red]"); raise typer.Exit(code=1)
    if not spacy_doc: console.print("[bold red]Error: spaCy doc creation failed.[/bold red]"); raise typer.Exit(code=1)

    # --- Analizleri Çalıştır ---
    console.print("\n[bold blue]--- Running Analysis Modules ---[/bold blue]")
    global sentences
    sentences = [data_models.SentenceInfo(text=s.text, start_char=s.start_char, end_char=s.end_char, tokens=[t.text for t in s]) for s in spacy_doc.sents]

    console.print("[cyan]Running Argument Analyzer (Enhanced)...[/cyan]")
    argument_components = argument_analyzer.enhanced_component_analyzer(spacy_doc)

    console.print("[cyan]Running Logic Analyzer (Enhanced - Rules + ML)...[/cyan]")
    fallacy_findings = logic_analyzer.enhanced_fallacy_analyzer(spacy_doc)

    evidence_findings = [] # Kanıt analizi kaldırıldı

    console.print("[cyan]Running Rhetoric Analyzer...[/cyan]")
    rhetoric_findings = rhetoric_analyzer.simple_rhetoric_analyzer(spacy_doc)

    console.print("[cyan]Running Synthesis Engine (Evidence Excluded)...[/cyan]")
    all_findings: List[data_models.Finding] = fallacy_findings + rhetoric_findings
    analysis_summary = synthesis_engine.generate_summary_ratings(argument_components, all_findings)

    # --- Argüman Grafiğini Oluştur (Embeddingler ile) ---
    console.print("[cyan]Building Argument Graph (Semantic Linking)...[/cyan]")
    # Görselleyiciye embedding listesini de gönder
    argument_graph = argument_visualizer.build_argument_graph(argument_components, sentence_embeddings) # <-- DEĞİŞİKLİK BURADA
    graph_text_representation = argument_visualizer.format_graph_text(argument_graph)

    # --- Sonuç Nesnesini Oluştur ---
    analysis_result = data_models.AnalyzedText(
        original_text=text_to_analyze, sentences=sentences,
        argument_components=argument_components, findings=all_findings,
        analysis_summary=analysis_summary
    )

    # --- Raporlama ---
    console.rule("[bold green]ETHOS Analysis Report[/bold green]", style="green")
    # Bölüm 1: Özet (Aynı)
    summary_table = Table(title="Analysis Summary", show_header=False, box=None, padding=(0, 1))
    # ... (Özet tablo kodu aynı) ...
    if analysis_result.analysis_summary:
        for category, rating in analysis_result.analysis_summary.items():
            style = "red" if rating.startswith(("Low", "Weak", "Questionable")) else "yellow" if rating.startswith(("Medium", "Moderate", "Mixed")) else "green"
            if rating == "Not Evaluated": style = "dim"
            summary_table.add_row(category, f"[{style}]{rating}[/{style}]")
    else: summary_table.add_row("Summary", "[dim]Not generated.[/dim]")
    console.print(Padding(summary_table, (1, 0)))


    # Bölüm 2: Tespit Edilen Bulgular (Aynı)
    console.print("\n[bold underline]Detected Findings:[/bold underline]")
    if not analysis_result.findings: console.print(Padding("  No significant findings detected.", (0, 2)))
    else:
        # ... (Bulgu gruplama ve yazdırma kodu aynı) ...
        grouped_findings = {};
        for f in analysis_result.findings: grouped_findings.setdefault(f.finding_type, []).append(f)
        grouped_findings.pop("EvidenceIndicator", None); grouped_findings.pop("EvidenceStatus", None)
        if not grouped_findings: console.print(Padding("  No significant findings detected.", (0, 2)))
        else:
            for f_type, findings_list in grouped_findings.items():
                 color = "red" if f_type=="Fallacy" else "yellow" if f_type=="RhetoricalDevice" else "white"
                 console.print(Padding(f"[bold {color}]{f_type} Indicators ({len(findings_list)} found):[/bold {color}]", (1, 1)))
                 for i, finding in enumerate(findings_list[:max_findings_display]):
                     details_dict = finding.details if finding.details else {}
                     details_text = details_dict.get('fallacy_type') or details_dict.get('device_type') or 'Details N/A'
                     trigger_text = details_dict.get('trigger') or details_dict.get('words')
                     confidence = details_dict.get('confidence'); model_used = details_dict.get('model_used')
                     details_str = f"({details_text}"
                     if trigger_text and not model_used: details_str += f", Trigger: '{textwrap.shorten(str(trigger_text), width=25, placeholder='...')}'"
                     if confidence is not None: details_str += f", Score: {confidence:.2f}"
                     if model_used: details_str += f", Model: '{model_used.split('/')[-1]}'"
                     details_str += ")"
                     console.print(Padding(f"{i+1}. {finding.description} {details_str}", (0, 3)))
                     try:
                          sentence_idx = next(idx for idx, s in enumerate(analysis_result.sentences) if s.start_char == finding.span_start)
                          related_sentence_text = analysis_result.sentences[sentence_idx].text
                          console.print(Padding(f"[dim]   In Sent {sentence_idx+1}: \"{textwrap.shorten(related_sentence_text, width=90, placeholder='...')}\"[/dim]", (0, 5)))
                     except StopIteration: console.print(Padding(f"[dim]   (Could not pinpoint exact sentence for span starting at char {finding.span_start})[/dim]", (0,5)))
                 if len(findings_list) > max_findings_display: console.print(Padding(f"... and {len(findings_list) - max_findings_display} more.", (0, 3)))


    # Bölüm 3: Argüman Bileşenleri (Aynı)
    console.print("\n[bold underline]Identified Argument Components:[/bold underline]")
    if not analysis_result.argument_components: console.print(Padding("  No argument components identified.", (0, 2)))
    else:
        # ... (Bileşen tablosu kodu aynı) ...
        comp_table = Table(title=None, show_header=True, header_style="bold magenta", box=None, padding=(0,1))
        comp_table.add_column("Type", style="magenta", width=10); comp_table.add_column("Text Snippet (Confidence)")
        for comp in analysis_result.argument_components:
             conf_str = f"({comp.confidence:.2f})" if comp.confidence is not None else ""
             comp_table.add_row(comp.component_type, f"\"{textwrap.shorten(comp.text, width=90, placeholder='...')}\" {conf_str}")
        console.print(Padding(comp_table, (1, 1)))


    # Bölüm 4: Argüman Yapısı Görselleştirmesi (Güncellenmiş formatı yazdıracak)
    console.print("\n[bold underline]Argument Structure Visualization (Semantic):[/bold underline]") # Başlık güncellendi
    console.print(Padding(graph_text_representation, (0, 1))) # <-- GRAFİĞİ YAZDIR
    # Not metni güncellendi
    console.print(Padding(f"[dim](Note: Links based on semantic similarity >= {argument_visualizer.LINKING_SIMILARITY_THRESHOLD})[/dim]", (0,1)))


    console.rule(style="green")
    console.print(f"(V2 Semantic Argument Linking. Total potential findings: {len(analysis_result.findings)})") # Mesaj güncellendi


# Script doğrudan çalıştırıldığında typer.run ile main fonksiyonunu çağır
if __name__ == "__main__":
    typer.run(main)