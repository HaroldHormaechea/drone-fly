# UC-01 test fixture provenance

- **Source dataset:** MaleCNS (mcns_inprop_all_neuron)
- **Source matrix URL:** https://raw.githubusercontent.com/YijieYin/connectome_data_prep/main/data/maleCNS/mcns_inprop_all_neuron.npz
- **Source meta URL:** https://raw.githubusercontent.com/YijieYin/connectome_data_prep/main/data/maleCNS/mcns_all_neuron_meta.csv
- **Source scale:** 161429 neurons, 25083972 edges (stored non-zeros)
- **Slice rule:** top-300 neurons by total degree (in-degree + out-degree,
  counted as stored non-zeros), tie-broken by ascending matrix index; induced
  submatrix over exactly those neurons. Fully deterministic — no randomness.
- **Resulting fixture scale:** 300 neurons, 10600 edges.

## Attribution

Derived from the connectome_data_prep dataset (https://github.com/YijieYin/connectome_data_prep), which packages the MaleCNS connectome (Janelia FlyEM / neuPrint). MaleCNS is released under CC-BY; this fixture is a small deterministic slice redistributed with attribution.
