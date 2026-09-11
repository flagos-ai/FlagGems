"""Stable Pretune artifact schemas and serialization helpers.

Reporting code reads shape YAML and writes CSV, JSONL, manifests, and logs; it
does not select GPUs, invoke operators, or train ranking models.
"""
