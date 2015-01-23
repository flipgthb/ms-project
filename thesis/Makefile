
FILE = document
IGNORED = /home/felippe/.aspell.pt_BR.pws
all: $(FILE).pdf run chapters/chapter*.tex $(FILE).bib graphics/*

$(FILE).pdf : $(FILE).tex ./chapters/chapter*.tex $(FILE).bib style.sty graphics/*
	latexmk -pdf -bibtex $<
	latexmk -c

chapters/chapter*.tex:
	aspell --lang=pt_BR -t -c $@
	aspell --lang=en -c $(IGNORED)
	aspell --lang=pt_BR -t -c $@

run:
	evince $(FILE).pdf &
	clear
