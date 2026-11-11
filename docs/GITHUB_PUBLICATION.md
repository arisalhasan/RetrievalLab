# GitHub publication

This public repository is https://github.com/arisalhasan/RetrievalLab.

The repository's `.gitattributes` disables automatic line-ending conversion. This preserves the exact source/configuration/experiment bytes required by the recorded SHA-256 checksums when cloning on Windows or another platform.

The versioned SciFact study was completed locally before publication. Original source modules, configurations, model-selection rule, split identities, metric records, and freeze metadata are preserved. The README adds a GitHub-facing results summary, and the Streamlit theme is included. Generated model weights, raw data, embeddings, and Python environments are excluded from Git history. Download the pinned pretrained assets or run the documented optional fine-tuning recipe before requesting neural methods.

The original release freeze expects the original trained checkpoint and exact recorded environment. A new machine or newly trained checkpoint must not impersonate that original experiment. Preserve a new study's provenance and disclose that the official test has already been inspected. All original data-overlap and performance limitations remain documented.

Historical verification reports describe the local release checkpoint. Live repository checks are shown in GitHub Actions; no remote success is assumed in those historical reports. The workflow's ordinary suite stays offline; real downloaded-data/model checks remain explicit opt-ins.

The first remote install exposed the hosted runner's preinstalled `pipx` dependency conflicting with the experiment's pinned `packaging` version. CI therefore creates a clean virtual environment and runs every install/check/test through that environment, matching the documented local setup. This publication-only pipeline correction does not change benchmark dependencies, model code, configurations or results.

The source is published without assigning a new project license. Upstream data/model terms remain those of their respective providers; no third-party pretrained or trained weight files are redistributed in this repository.
