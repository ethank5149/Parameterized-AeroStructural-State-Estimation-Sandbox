FILENAME = template

all: $(FILENAME).pdf

$(FILENAME).pdf: $(FILENAME).tex $(FILENAME).bib
	pdflatex $(FILENAME)
	bibtex $(FILENAME)
	pdflatex $(FILENAME)
	pdflatex $(FILENAME)

clean:
	rm -f *.aux *.log *.out *.pdf *.bbl *.blg *.toc *.fdb_latexmk *.fls
