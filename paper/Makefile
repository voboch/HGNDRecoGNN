# Minimal LaTeX build for the HGND + GNN sensitivity paper skeleton.
# Run `make` for a single-file build; `make bib` runs bibtex too.

TEX = pdflatex -interaction=nonstopmode -halt-on-error
DOC = main

.PHONY: all bib clean distclean

all: $(DOC).pdf

$(DOC).pdf: $(DOC).tex
	$(TEX) $(DOC).tex
	$(TEX) $(DOC).tex

bib: $(DOC).tex
	$(TEX) $(DOC).tex
	bibtex $(DOC)
	$(TEX) $(DOC).tex
	$(TEX) $(DOC).tex

clean:
	rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc \
	      $(DOC).bbl $(DOC).blg $(DOC).fdb_latexmk \
	      $(DOC).fls $(DOC).synctex.gz

distclean: clean
	rm -f $(DOC).pdf
