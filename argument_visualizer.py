# argument_visualizer.py (V2/V3 - Adım 3.1: Düşük Eşik Denemesi)
import networkx as nx
from typing import List, Dict
import data_models
from rich.console import Console
import textwrap
import torch
import torch.nn.functional as F

console = Console()

# Anlamsal Benzerlik Eşiği DÜŞÜRÜLDÜ!
LINKING_SIMILARITY_THRESHOLD = 0.55 # Önceki 0.65 idi, şimdi 0.55 deneyelim

# --- Yardımcı Fonksiyon: Benzerlik Hesaplama (Aynı) ---
def calculate_similarity(emb1: torch.Tensor | None, emb2: torch.Tensor | None) -> float | None:
     if emb1 is None or emb2 is None: return None
     emb1 = emb1.cpu(); emb2 = emb2.cpu()
     if emb1.dim() == 1: emb1 = emb1.unsqueeze(0)
     if emb2.dim() == 1: emb2 = emb2.unsqueeze(0)
     try: return F.cosine_similarity(emb1, emb2).item()
     except Exception as e: console.print(f"[yellow]Warn: Cosine similarity failed: {e}[/yellow]"); return None

# --- Ana Grafik Oluşturma Fonksiyonu (Aynı) ---
def build_argument_graph(
    components: List[data_models.ArgumentComponent],
    sentence_embeddings: List[torch.Tensor | None]
) -> nx.DiGraph | None:
    # ... (Fonksiyonun geri kalanı bir önceki mesajdakiyle aynı)...
    if not components: return None
    G = nx.DiGraph()
    claims = []; premises = []
    for i, comp in enumerate(components):
        node_id = f"{comp.component_type[0]}{i+1}"
        conf_str = f" (Conf: {comp.confidence:.2f})" if comp.confidence is not None else ""
        node_label = f"\"{textwrap.shorten(comp.text, width=35, placeholder='...')}\"{conf_str}"
        node_shape = "()" if comp.component_type == "Claim" else "[]"
        G.add_node(node_id, label=node_label, type=comp.component_type, shape=node_shape, component_index=i, sentence_index=comp.sentence_index)
        if comp.component_type == "Claim": claims.append((i, node_id, comp.sentence_index))
        else: premises.append((i, node_id, comp.sentence_index))

    if not claims or not premises:
        console.print(" -> Argument graph requires at least one claim and one premise to show structure.", style="dim")
        return G

    edges_added = 0
    for p_comp_idx, p_node_id, p_sent_idx in premises:
        if not (0 <= p_sent_idx < len(sentence_embeddings)): continue
        premise_embedding = sentence_embeddings[p_sent_idx]
        if premise_embedding is None: continue
        for c_comp_idx, c_node_id, c_sent_idx in claims:
             if not (0 <= c_sent_idx < len(sentence_embeddings)): continue
             claim_embedding = sentence_embeddings[c_sent_idx]
             if claim_embedding is None: continue
             similarity = calculate_similarity(premise_embedding, claim_embedding)
             if similarity is not None and similarity >= LINKING_SIMILARITY_THRESHOLD: # Eşik kontrolü
                  console.print(f"   -> Linking {p_node_id} to {c_node_id} (Similarity: {similarity:.2f})", style="dim")
                  G.add_edge(p_node_id, c_node_id, relation="supports", similarity=similarity)
                  edges_added += 1
    if edges_added == 0:
         console.print(f" -> Could not link any premises to claims based on similarity threshold ({LINKING_SIMILARITY_THRESHOLD}).", style="dim")
    console.print(f" -> Built argument graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges (Semantic Linking).", style="dim")
    return G


# Grafik Formatlama Fonksiyonu (Aynı)
def format_graph_text(graph: nx.DiGraph | None) -> str:
    # ... (Fonksiyonun içeriği öncekiyle aynı) ...
    if graph is None: return "  Argument graph could not be built."
    if graph.number_of_nodes() == 0: return "  No argument components identified to build a graph."
    output_lines = ["  Argument Structure (Linked by Semantic Similarity):"]
    isolated_nodes = [node for node, degree in graph.degree() if degree == 0 and graph.in_degree(node)==0 and graph.out_degree(node)==0 ]
    connected_nodes = set()
    if graph.number_of_edges() == 0:
        if not isolated_nodes: output_lines.append("  No argument structure links could be determined.")
    else:
        output_lines.append("  Detected Links (Premise --Supports (Similarity)--> Claim):")
        for u, v, data in graph.edges(data=True):
            connected_nodes.add(u); connected_nodes.add(v)
            u_data = graph.nodes[u]; v_data = graph.nodes[v]
            u_shape = u_data.get('shape', '[]'); v_shape = v_data.get('shape', '()')
            u_label = u_data.get('label', u); v_label = v_data.get('label', v)
            similarity = data.get('similarity')
            sim_str = f"(Sim: {similarity:.2f})" if similarity is not None else ""
            output_lines.append(f"    {u_shape[0]}{u}{u_shape[1]}: {u_label}")
            output_lines.append(f"      {' ' * len(u)} --Supports {sim_str}--> {v_shape[0]}{v}{v_shape[1]}: {v_label}")
            output_lines.append("")

    really_isolated = set(isolated_nodes) # Tüm izoleler
    if really_isolated:
         # Eğer kenar varsa ve izole düğüm varsa başlık ekle
         if graph.number_of_edges() > 0:
             output_lines.append("  Nodes without determined links:")
         # Kenar yoksa zaten yukarıda başlık vardı
         elif not isolated_nodes and graph.number_of_edges() == 0 : # Hata durumu, yukarıda handle edildi
             pass
         elif isolated_nodes and graph.number_of_edges() == 0: # Sadece izole düğümler varsa
              output_lines.append("  Nodes (No links determined):")


         for node_id in sorted(list(really_isolated)):
              data = graph.nodes[node_id]
              shape = data.get('shape', '??')
              label = data.get('label', node_id)
              output_lines.append(f"    {shape[0]}{node_id}{shape[1]}: {label}")

    output_lines.append(f"  [dim](Links based on similarity >= {LINKING_SIMILARITY_THRESHOLD})[/dim]")
    return "\n".join(output_lines)