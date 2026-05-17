# Datasets

This package will own the train-ready data representation for the new pipeline.

Planned responsibilities:

- validate canonical trajectory-prediction sample structure
- convert episode-frame exports into model-ready samples
- store sample tables and split metadata
- store normalization statistics for reproducible training

The current canonical schema is defined in:

- [schema.py](./schema.py)

