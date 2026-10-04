"""sargolini-toolkit -- load the Sargolini et al. 2006 recordings and reproduce
their grid / head-direction / conjunctive classification.

    Sargolini F, Fyhn M, Hafting T, McNaughton BL, Witter MP, Moser M-B,
    Moser EI (2006). Conjunctive representation of position, direction, and
    velocity in entorhinal cortex. Science 312(5774): 758.
    doi:10.1126/science.1125572

Data: https://doi.org/10.11582/2014.00003 (not included; see
scripts/download_data.py).

`data_io` loads and normalises sessions, `cells` computes per-cell scores and
the circular-shift null, `run_classify` applies the thresholds.
"""

__version__ = "0.1.0"
