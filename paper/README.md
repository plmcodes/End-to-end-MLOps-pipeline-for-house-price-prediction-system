# Term paper

- `Term_Paper_House_Price_Prediction.docx` / `.pdf`: the paper.
- `build_paper.js`: regenerates the .docx from the numbers and figures in `research/`.

Rebuild (run from the repo root, Node.js required):

```
npm install docx@9.5.1
node paper/build_paper.js . paper/Term_Paper_House_Price_Prediction.docx
```

Rerun the scripts in `research/` first if the data or code change.
