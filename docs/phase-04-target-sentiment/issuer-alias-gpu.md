# Reviewed aliases on the GPU PC

The verified alias release uses the project's existing RTX 4070 SUPER environment,
NER weights and saved target-aware RoBERTa checkpoint. Run from PowerShell:

```powershell
cd C:\Hope\Project\financial-news-intelligence

.\finsight-organizations.cmd --text "Toyota reported higher profits. Boeing reported a loss. Dell announced higher sales." --company-context --with-sentiment --device cuda
```

The organization launcher now selects the pinned SEC directory and reviewed
alias catalog automatically. `--confirmed-company` is unnecessary for these
reviewed names under explicit company context. Other names still need evidence;
this is not a new trained company classifier. NER still must detect each name.
`--company-context` remains your explicit assertion, not a silent launcher default.

Expected identity behavior:

- Toyota links to Toyota Motor Corporation; Boeing links to The Boeing Company.
- Dell is a company and receives sentiment, but its listed parent stays unresolved.
- Add `--allow-parent-rollup` to permit Dell's reviewed parent mapping:

```powershell
.\finsight-organizations.cmd --text "Dell announced higher sales." --company-context --allow-parent-rollup --with-sentiment --device cuda
```

Parent roll-up changes identity/grouping eligibility only. It does not replace
“Dell” with the parent name in model input or change the target sentiment score.
A link does not establish historical ownership or security-type validity.

Direct targets still work:

```powershell
.\finsight-predict.cmd --text "Toyota reported higher profits." --target Toyota --device cuda
```

Both launchers accept `--help`. The underlying organization module still requires
explicit `--us-directory` and `--company-aliases` when invoked without the launcher;
the launcher is what supplies those pinned paths. The model input remains the
original target/headline pair, with no masking. Probabilities are uncalibrated.

## Installed release and rollback

The selected source manifest is
`f36459ecf9743b434df509b84963aa676bc135d4d025dcfc173758373ecc264a` under
`data\interim\organization-release`. The launchers set `PYTHONPATH` for their
process, run the existing `.venv\Scripts\python.exe` and use shared data/model
paths. They require no `uv` and run offline. The ordinary tracked checkout/Git
branch was preserved, so plain Python module invocation without selecting the
release still uses the older checkout.

The prior source release and its exact launcher bytes are retained. To restore
both prior launchers from the project root:

```powershell
Copy-Item .\data\interim\organization-release\f36459ecf9743b434df509b84963aa676bc135d4d025dcfc173758373ecc264a\previous-launchers\finsight-*.cmd -Destination .
```

Keep the selected release directory because the launchers point to it. The small
transfer ZIP was deleted after verified installation. No model weights, news
records, approved references, contacts or `.env` were copied. The exact smoke
script, manifest and GPU receipt are retained inside the release.

[CUDA verification and provenance](../../reports/phase-04-target-sentiment/verification/issuer-alias-gpu.md).
