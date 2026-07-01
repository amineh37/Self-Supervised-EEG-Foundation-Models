# STELAR — Architecture report

LaTeX source for the STELAR architecture report.

- `report.tex` — the report (self-contained `article` class; bibliography embedded
  via `thebibliography`, so no BibTeX/Biber pass is needed).
- `stelar_arch_fig.png` — architecture figure included by the document.

## Build

With the provided Makefile:

```bash
make          # builds report.pdf
make clean    # removes LaTeX build artifacts
make distclean# also removes report.pdf
make watch    # continuous rebuild (needs latexmk)
```

The Makefile uses `latexmk` when available, otherwise falls back to two
`pdflatex` passes (needed to resolve `hyperref`/`cleveref` references).

Manual build without make:

```bash
latexmk -pdf report.tex            # recommended
# or:
pdflatex report.tex && pdflatex report.tex
```

## Requirements

A TeX Live (or MiKTeX) install providing `pdflatex` and the packages used by the
document: `geometry`, `microtype`, `amsmath`, `bm`, `graphicx`, `booktabs`,
`tikz`, `algorithm2e`, `hyperref`, `cleveref`, `titlesec`. `latexmk` is
recommended but optional.

Output: `report.pdf`.
