"""
cache_latents.py
────────────────
Precomputes and saves [N, 256] latents for all 6 models across all 5 stocks.
Saves: latents/{model_name}/{stock}/{train,val,test}_latents.npy
       latents/{model_name}/{stock}/{train,val,test}_labels.npy
       latents/{model_name}/{stock}/theta.npy

Usage on Colab:
    !python cache_latents.py
"""

import os
import sys
import time
import torch
from downstream_common import (
    MODEL_REGISTRY, STOCKS, set_seed, precompute_and_cache_latents
)

def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 80)
    print(f"  PHASE 2: PRECOMPUTING LATENT EMBEDDINGS (Device: {device})")
    print("=" * 80)

    total_start = time.time()
    count = 0
    total = len(MODEL_REGISTRY) * len(STOCKS)

    for model_name in MODEL_REGISTRY.keys():
        print(f"\n[{model_name}]")
        for stock in STOCKS:
            count += 1
            print(f"({count}/{total}) Caching {model_name} / {stock} ...")
            t0 = time.time()
            try:
                precompute_and_cache_latents(model_name, stock, out_dir="latents", device=device)
                print(f"    Done in {time.time() - t0:.1f}s")
            except Exception as e:
                print(f"    ❌ ERROR caching {model_name}/{stock}: {e}")

    print("\n" + "=" * 80)
    print(f"  LATENT PRECOMPUTATION COMPLETE ({time.time() - total_start:.1f}s)")
    print("=" * 80)

if __name__ == '__main__':
    main()
