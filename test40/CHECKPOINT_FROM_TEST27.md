# Test30 Checkpoint

This folder was created on 2026-04-02 as a local working checkpoint cloned from `test27`.

Intent:
- freeze `test27` as the completed phase checkpoint
- start `test30` as the next local development version

Included:
- current Python source files
- runner scripts and local Ollama bootstrap scripts
- documentation and requirements files
- reusable `entities/` inputs carried over from `test27`
- critic prompt assets

Intentionally excluded from the copy:
- generated `.arrow` outputs
- graph/export/runtime artifact folders
- logs, checkpoints, caches, and scratch files
- generated planning JSON artifacts at the repo root

If needed, regenerate runtime outputs in `test30` by running the pipeline fresh.
