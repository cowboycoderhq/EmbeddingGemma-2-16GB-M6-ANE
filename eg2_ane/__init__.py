"""EmbeddingGemma 2 (text) on the Apple Neural Engine (Core AI), for 16 GB M6 Macs.

Modules:
  model      the sentence-transformers text model rewritten in the ANE layout, with the FP16 and Core AI fixes
  build      PyTorch -> one .aimodel with several entry points (one weight copy), FP16 weights
  runtime    load the .aimodel and embed text (host-side token lookup, batching, padding)
  check      quality and speed checks against the FP32 sentence-transformers model
  reference  make the FP32 reference data for `check quality`
  retrieval  SciFact retrieval check (nDCG@10) for any engine
  release    write release.json (size and sha256 of every model file)
"""

__version__ = "0.1.0"
