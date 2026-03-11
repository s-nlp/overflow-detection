import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from metrics import (  
    compute_group_saturation_metrics,
)
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
XRAG_DIR = os.path.abspath(os.path.join(HERE, "..", "xRAG"))

sys.path.insert(0, XRAG_DIR)
# from src.model import XGemmaForCausalLM, SFR
from src.model import XMistralForCausalLM, SFR
from src.language_modeling.utils import XRAG_TOKEN, get_retrieval_embeds


rag_template_mistral = """<s>[INST] Refer to the background document and answer the questions:

Background: {document}

Question: {question} [/INST] The answer is:"""

def prompt_xrag_mistral(question: str, n_segments: int) -> str:
    return rag_template_mistral.format(
        document=" ".join([XRAG_TOKEN] * n_segments),
        question=question,
    )

def format_one_example(
    sample,
    use_rag,
    retrieval_embed_length,
    task_type="open_qa",
):
    question = sample["question"].strip()
    backgrounds = sample.get("background", []) if use_rag else []

    if use_rag:
        if retrieval_embed_length > 0:
            # XRAG token document placeholder
            prompt = prompt_xrag_mistral(question=question, n_segments=retrieval_embed_length)
        else:
            # (rare) fall back to raw background in "document"
            background_text = "\n\n".join(backgrounds)
            prompt = rag_template.format(document=background_text, question=question) + ""
        return prompt, backgrounds

    return prompt, backgrounds
    
def extract_vector(t: torch.Tensor, name: str, mode: str = "first") -> torch.Tensor:
    """
    Robustly extract a 1D vector from t.

    Accepts tensors with shapes like:
      [D], [L, D], [B, L, D], [1, L, D], [1, 1, L, D], etc.
    Policy:
      - collapse leading singleton dims (while preserving last dim = feature dim)
      - if resulting tensor is 2D ([L, D]) -> either take first row (mode="first") or mean across L (mode="mean")
      - if 1D -> return it
    Returns: 1D torch.Tensor on CPU (float32)
    """
    if t is None:
        raise RuntimeError(f"{name} is None (hook may not have fired / wrong module).")

    # detach + move to CPU early
    if isinstance(t, torch.Tensor):
        t_cpu = t.detach().cpu()
    else:
        raise RuntimeError(f"{name} expected torch.Tensor, got {type(t)}")

    # collapse leading singleton dims until we have at most 3 dims or until last dim looks like features
    # e.g. (1,1,1,4096) -> (1,1,4096) -> (1,4096) -> [4096]
    while t_cpu.ndim > 1 and t_cpu.shape[0] == 1:
        t_cpu = t_cpu[0]

    # now possible shapes: [D], [L, D], [B?, L, D] (but collapsed leading singletons so B=1 removed)
    if t_cpu.ndim == 1:
        vec = t_cpu
    elif t_cpu.ndim == 2:
        # [L, D] -> choose policy
        if mode == "first":
            vec = t_cpu[0]
        elif mode == "mean":
            vec = t_cpu.mean(dim=0)
        else:
            raise ValueError("mode must be 'first' or 'mean'")
    else:
        # unexpected higher-rank tensor (shouldn't happen after collapse)
        raise RuntimeError(f"{name} expected 1D or 2D after collapsing singletons; got shape {tuple(t_cpu.shape)}")

    if vec.ndim != 1:
        raise RuntimeError(f"{name} extraction failed, result not 1D: {tuple(vec.shape)}")

    return vec.to(torch.float32)
    
# Capture post projection 
class CaptureHook:
    def __init__(self):
        self.value = None

    def __call__(self, module, inputs, output):
        # output is usually [B, embed_len, d_model] or [embed_len, d_model]
        self.value = output.detach().cpu()

def get_xrag_states_with_projection(
    sample: dict,
    model,
    tokenizer,
    # retriever,
    # retriever_tokenizer,
    ctx2embed,
    mid_layer_index: int = 13,
    task_type: str = "open_qa",
    use_rag: bool = True,
    splitter: str = "\n\n",
    device: str = "cuda:0",
    debug: bool = False,
):
    # retrieval_embed_length = retriever.get_embed_length()
    retrieval_embed_length = 1

    _, backgrounds = format_one_example(
        sample,
        use_rag=use_rag,
        retrieval_embed_length=retrieval_embed_length,
        task_type=task_type,
    )
    # background_text = "\n\n".join(backgrounds)

    cid = str(sample["id"])
    emb_list = ctx2embed[cid]

    retrieval_embeds = torch.tensor(
        emb_list,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    if XRAG_TOKEN in tokenizer.get_vocab():
        xrag_token_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
        model.set_xrag_token_id(xrag_token_id)
    else:
        raise ValueError(f"{XRAG_TOKEN} not found in tokenizer vocabulary.")
    
    # Mistral prompt
    prompt = prompt_xrag_mistral(sample["question"].strip(), retrieval_embed_length)
    
    # Find the first token before "[/INST]" by tokenizing the text up to that point
    # This works with any tokenizer (Python or Fast)
    inst_close_pos = prompt.find("[/INST]")
    if inst_close_pos == -1:
        raise ValueError("[/INST] not found in prompt")
    
    # Tokenize the prompt up to (but not including) "[/INST]"
    prompt_before_inst = prompt[:inst_close_pos]
    tokens_before_inst = tokenizer(prompt_before_inst, add_special_tokens=False)["input_ids"]
    # The last token before "[/INST]" is the last token in this tokenization
    inst_token_idx = len(tokens_before_inst) - 1
    
    if inst_token_idx < 0:
        raise ValueError("Could not find token before [/INST]")
    
    # Create inputs for the full prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False, add_special_tokens=False)
    
    if debug:
        # Validation: Print information about the found token
        print(f"[VALIDATION] Finding token before [/INST] in xRAG prompt:")
        print(f"  Prompt snippet: ...{prompt[max(0, inst_close_pos-50):inst_close_pos+10]}...")
        print(f"  [/INST] position: {inst_close_pos}")
        print(f"  Token index before [/INST]: {inst_token_idx}")
        # Decode tokens around the found position for validation
        all_tokens = inputs["input_ids"][0].tolist()
        context_start = max(0, inst_token_idx - 2)
        context_end = min(len(all_tokens), inst_token_idx + 5)
        context_tokens = all_tokens[context_start:context_end]
        context_text = tokenizer.decode(context_tokens)
        print(f"  Tokens around position {inst_token_idx}: {context_tokens}")
        print(f"  Decoded context: {context_text}")
        print(f"  Token at position {inst_token_idx}: '{tokenizer.decode([all_tokens[inst_token_idx]])}'")
        print()
    
    cap = CaptureHook()
    hook_handle = model.projector.register_forward_hook(cap)

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
                retrieval_embeds=retrieval_embeds.to(device),
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        hook_handle.remove()

    token_indices = (inputs.input_ids == xrag_token_id).nonzero(as_tuple=True)[1]
    # Create tensor with the token index before "[/INST]"
    inst_token_indices = torch.tensor([inst_token_idx], device=inputs.input_ids.device)

    if token_indices.numel() == 0:
        raise RuntimeError("No XRAG tokens found in the prompt input_ids.")

    hs = outputs.hidden_states
    last_layer_index = len(hs) - 1
    if mid_layer_index < 0 or mid_layer_index > last_layer_index:
        raise ValueError(f"mid_layer_index={mid_layer_index} out of range (0..{last_layer_index})")

    mid_h = hs[mid_layer_index][0]
    last_h = hs[last_layer_index][0]

    mid_xrag = mid_h[token_indices].detach().cpu()
    last_xrag = last_h[token_indices].detach().cpu()

    mid_q = mid_h[inst_token_indices].detach().cpu()
    last_q = last_h[inst_token_indices].detach().cpu()

    mid_group = compute_group_saturation_metrics(mid_h, token_indices)
    last_group = compute_group_saturation_metrics(last_h, token_indices)

    proj_out = cap.value

    return {
        "id": sample.get("id", None),
        "token_indices": token_indices.detach().cpu(),
        "input_len": int(inputs.input_ids.shape[1]),
        "retrieval_embeds_preproj": retrieval_embeds.detach().cpu(),
        "proj_out_postproj": proj_out,
        "mid_layer_index": mid_layer_index,
        "last_layer_index": last_layer_index,
        "mid_xrag": mid_xrag,
        "last_xrag": last_xrag,
        
        "mid_q": mid_q,
        "last_q": last_q,

        # "mid_q_only": mid_q_only,
        # "last_q_only": last_q_only,

        "mid_group_metrics": mid_group,
        "last_group_metrics": last_group, # context_last_group,  # Use context group metrics instead of xRAG
    }


@torch.no_grad()
def get_xrag_attention_stats(
    sample: dict,
    model,
    tokenizer,
    # retriever,
    # retriever_tokenizer,
    ctx2embed: dict,                 # your cached embeds dict
    layer_indices=None,              # e.g. [0, 8, 16, -1] or None=all layers
    device: str = "cuda",
    max_length_retriever: int = 180,
):
    """
    Returns attention-based metrics comparing XRAG tokens vs non-XRAG tokens.
    Works for Mistral xRAG prompt.

    Metrics per (layer, head):
      - xrag_out_to_nonxrag_mean: mean attention mass from XRAG query positions to non-XRAG keys
      - xrag_out_to_xrag_mean:    mean attention mass from XRAG queries to XRAG keys
      - nonxrag_in_to_xrag_mean:  mean attention mass from non-XRAG queries to XRAG keys
      - xrag_in_share:            fraction of total incoming attention going to XRAG keys
      - xrag_out_entropy:         entropy of XRAG query attention distribution (over keys), averaged

    Note: we aggregate over all XRAG positions (if multiple).
    """

    # --- build retrieval_embeds from cached embedding ---
    cid = str(sample["id"])
    if cid not in ctx2embed:
        raise KeyError(f"id={cid} not found in ctx2embed")
    emb_list = ctx2embed[cid]  # [H]
    retrieval_embeds = torch.tensor(emb_list, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H]

    # --- prompt ---
    # n_segments = retriever.get_embed_length()
    n_segments = 1
    prompt = prompt_xrag_mistral(sample["question"].strip(), n_segments)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False, add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # --- XRAG token id and positions ---
    if XRAG_TOKEN not in tokenizer.get_vocab():
        raise ValueError(f"{XRAG_TOKEN} not found in tokenizer vocabulary.")
    xrag_token_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    model.set_xrag_token_id(xrag_token_id)

    input_ids = inputs["input_ids"][0]
    xrag_pos = (input_ids == xrag_token_id).nonzero(as_tuple=True)[0]  # [n_xrag]
    if xrag_pos.numel() == 0:
        raise RuntimeError("No XRAG tokens found in input_ids")

    seq_len = input_ids.shape[0]
    all_pos = torch.arange(seq_len, device=device)
    nonxrag_pos = all_pos[~torch.isin(all_pos, xrag_pos)]

    # --- forward with attentions ---
    outputs = model(
        **inputs,
        retrieval_embeds=retrieval_embeds.to(device),
        output_attentions=True,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )

    attentions = outputs.attentions
    # usually: tuple length = n_layers, each [B, n_heads, S, S]
    n_layers = len(attentions)

    if layer_indices is None:
        layer_indices = list(range(n_layers))
    else:
        # allow negatives
        layer_indices = [(li if li >= 0 else n_layers + li) for li in layer_indices]

    def entropy(p, eps=1e-12):
        p = torch.clamp(p, min=eps)
        return -(p * p.log()).sum(dim=-1)

    stats = []
    for li in layer_indices:
        A = attentions[li][0]  # [H, S, S]
        n_heads = A.shape[0]

        # XRAG queries => keys (outgoing from XRAG tokens)
        A_xq = A[:, xrag_pos, :]                 # [H, n_xrag, S]
        out_to_xrag = A_xq[:, :, xrag_pos].mean(dim=(1,2))          # [H]
        out_to_nonx = A_xq[:, :, nonxrag_pos].mean(dim=(1,2))       # [H]

        # Non-XRAG queries => XRAG keys (incoming to XRAG tokens)
        A_nq = A[:, nonxrag_pos, :]              # [H, n_nonx, S]
        in_from_nonx = A_nq[:, :, xrag_pos].mean(dim=(1,2))         # [H]

        # XRAG incoming share: sum attention mass going to XRAG keys / total mass (over keys), for each head
        # Here we average over all query positions (all tokens)
        total_to_xrag = A[:, :, xrag_pos].sum(dim=-1).mean(dim=-1)  # [H]  (avg over queries)
        # total attention mass per query sums to 1, so sum over keys =1; mean over queries -> 1
        # But we used sum over xrag keys, so it's already a "share".
        xrag_in_share = total_to_xrag  # [H]

        # Entropy of XRAG outgoing attention distribution (over keys), averaged over XRAG queries
        # A_xq: [H, n_xrag, S] => entropy over last dim => [H, n_xrag] => mean => [H]
        xrag_out_ent = entropy(A_xq).mean(dim=1)  # [H]

        for h in range(n_heads):
            stats.append({
                "layer": li,
                "head": h,
                "xrag_out_to_xrag_mean": float(out_to_xrag[h].detach().cpu()),
                "xrag_out_to_nonxrag_mean": float(out_to_nonx[h].detach().cpu()),
                "nonxrag_in_to_xrag_mean": float(in_from_nonx[h].detach().cpu()),
                "xrag_in_share": float(xrag_in_share[h].detach().cpu()),
                "xrag_out_entropy": float(xrag_out_ent[h].detach().cpu()),
                "seq_len": int(seq_len),
                "n_xrag": int(xrag_pos.numel()),
            })

    return stats