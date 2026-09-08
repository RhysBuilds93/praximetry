# Architecture diagrams

Excalidraw source. Open at [excalidraw.com](https://excalidraw.com) (File → Open) or
the VS Code Excalidraw extension.

- `01-system-oss-cloud` — OSS package ↔ praximetry-cloud, one-way dependency, the
  `apply` → `overrides.json` → `_apply_overrides` return path.
- `02-recording-path` — SDK-patch / manual / OTel producers converging on the store.
  OTel writes straight to `store.save_call`, bypassing `record_call` (PRA-64).
- `03-eval-capture-flow` — `capture_request()` runs real `@px.stage` code up to its
  first outbound LLM call, then raises before any network contact.

Regenerate after code changes: `.venv/bin/python docs/diagrams/gen_excalidraw.py`.
Labels are grounded in the code as of the commit noted in the generator docstring.
