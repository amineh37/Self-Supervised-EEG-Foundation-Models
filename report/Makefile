MAIN    := report
TEX     := $(MAIN).tex
PDF     := $(MAIN).pdf

LATEXMK := $(shell command -v latexmk 2>/dev/null)

.PHONY: all pdf watch clean distclean

all: pdf

pdf: $(PDF)

ifeq ($(LATEXMK),)
$(PDF): $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)
else
$(PDF): $(TEX)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)
endif

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(TEX)

clean:
	-latexmk -c $(TEX) 2>/dev/null || true
	rm -f $(MAIN).aux $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).out \
	      $(MAIN).toc $(MAIN).bbl $(MAIN).blg $(MAIN).synctex.gz build.log

distclean: clean
	rm -f $(PDF)
