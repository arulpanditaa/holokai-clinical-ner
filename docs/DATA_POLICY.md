# Public snapshot and dataset handling

The source project works on clinical/privacy named entity recognition, including label types for people, identifiers, healthcare providers, medication and diagnoses.

This public-ready archive **does not contain** raw annotated train/validation/V4 records or per-example error/prediction dumps. An inspected source validation example included patient-like names and medical record numbers. This does not establish whether the original data was synthetic, but without documented source-level clearance it would be inappropriate to republish those records automatically in a public portfolio repository.

The `data/demo_gliner.json` sample is newly written, entirely fictional and contains no information copied from project records.

Also excluded: large training/model checkpoint binaries, derived tokenizers, transient logs, caches, and the original sponsor slide deck explicitly marked "Proprietary & Confidential." This is a disclosure-limited GitHub snapshot, **not** a complete mirror of the entire Google Drive hierarchy.

To reproduce an experiment, obtain the approved project data separately, verify its permissible use and source license, place it in a nontracked directory such as `local_data/`, and update the config paths accordingly. Do not commit local datasets or credentials. Do not assume that being accessible from a shared Drive implies permission for worldwide GitHub redistribution.

No new rights or authorization are asserted by this archive. Research reports remain attributed to the collaborative practicum. If the team approves release of a de-identified dataset and/or checkpoints, they can be distributed under an explicit documented license and, if required, Git LFS or a model registry.
