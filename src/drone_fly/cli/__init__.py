"""CLI stage: thin command-line entry points.

Orchestrates the pipeline stages: fetch-connectome, train, evaluate. Kept thin —
argument parsing and wiring only; the real logic lives in the stage subpackages.
"""
