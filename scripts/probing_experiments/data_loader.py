"""
Data loading utilities for probing experiments.
"""

import torch
from pathlib import Path
from typing import Dict, Any


def load_probing_data(data_path: str = "probe_out_mistral/vectors_probing.pt") -> Dict[str, torch.Tensor]:
    """
    Load probing data from .pt file.
    
    Expected keys:
    - 'ids': sample IDs
    - 'labels': overflow labels (binary)
    - 'preproj', 'postproj', 'mid', 'last': xRAG token embeddings
    - 'mid_q', 'last_q', 'mid_q_only', 'last_q_only': query token embeddings
    - 'preproj_q', 'postproj_q': query-related context embeddings
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, map_location='cpu')
    
    # Verify required keys
    required_keys = ['ids', 'labels']
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    
    # Convert to float32 if needed
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            if data[key].dtype in [torch.bfloat16, torch.float16]:
                data[key] = data[key].to(torch.float32)
    
    print(f"Loaded data with keys: {list(data.keys())}")
    print(f"Number of samples: {len(data['ids'])}")
    
    # Print shapes
    for key in ['preproj', 'postproj', 'mid', 'last', 'mid_q', 'last_q']:
        if key in data:
            print(f"  {key}: {data[key].shape}")
    
    return data

