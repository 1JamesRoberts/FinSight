# Run the verified company pipeline on the GPU PC

**Latest release:** The GPU launchers now include reviewed aliases and updated
source. Use [reviewed-alias GPU commands](issuer-alias-gpu.md). The source pin and
installation details below describe the preserved earlier release.

The current inference code is installed as a checksum-verified release inside
`C:\Hope\Project\financial-news-intelligence\data\interim\organization-release`.
It reuses that project's existing `.venv`, NER weights and RoBERTa deployment
weights. The SEC snapshot is installed under its ignored `data` directory.
The ordinary Git checkout was preserved; the two launchers select the new release
for their process through `PYTHONPATH`. They do not permanently alter your shell
or Python environment. No `uv` command is needed.

In PowerShell on the GPU PC:

```powershell
cd C:\Hope\Project\financial-news-intelligence

.\finsight-organizations.cmd --text "Toyota reported higher profits while Microsoft reported a loss." --confirmed-company Toyota --with-sentiment --device cuda
```

Toyota can receive sentiment while its company identity remains unresolved and
`aggregation_eligible` is false. Microsoft resolves under the existing registry.
Confirmation applies to the exact detected name, not all organizations, and does
not make NER invent a missing span. Probabilities remain uncalibrated model scores.

Direct targets still work through the updated release:

```powershell
.\finsight-predict.cmd --text "Toyota reported higher profits." --target Toyota --device cuda
```

For automatic SEC linking of a supported name:

```powershell
.\finsight-organizations.cmd --text "Tesla reported higher earnings." --us-directory .\data\interim\us-listed-directory\d76a4e6fd5d7cc540b3ef6974ece2ca508b3fd16360e8d9eca7b94ddce9f528b --company-context --with-sentiment --device cuda
```

`--company-context` supplies context for matching, not blanket company
classification. SEC associations preserve all listings and do not invent a
primary ticker. Unknown/uncertain organizations remain visible without default
sentiment, and empty results skip sentiment loading.

Both launchers accept `--help`. Plain `.venv\Scripts\python.exe -m ...` continues
to select the older checkout unless you explicitly select the new release via
`PYTHONPATH`; use these launchers for the verified pipeline. This task did not
commit, push or update that checkout's Git branch. The files are PC-local and
are not present in a fresh clone.

The pinned source manifest is
`3be40da98596b9b065358bbc01bb529d04db1720e1d8b7080163362698517924`.
Keep the selected release directory: the launchers point to it. The transferred
ZIP was deleted after successful installation; prior source releases are retained
as historical evidence. The manifest and GPU receipt are inside this release;
`smoke_gpu.py` records the exact invented-input verification procedure.
The source bundle contains only an explicit allowlist of Python files, required
public configuration/evidence, the small SEC snapshot and verification script.
It contains no `.env`, contact details, news records or model weights.

[GPU results and provenance](../../reports/phase-04-target-sentiment/verification/company-mention-routing-gpu.md).
