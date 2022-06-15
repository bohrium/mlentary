mlentary:
	pdflatex mlentary.tex
	pdflatex mlentary.tex
	evince 	 mlentary.pdf
	rm 		 mlentary.aux
	rm 		 mlentary.log
	rm 		 mlentary.out

rec-01:
	pdflatex rec-01.tex
	evince 	 rec-01.pdf
	rm 		 rec-01.aux
	rm 		 rec-01.log
	rm 		 rec-01.out

rec-02:
	pdflatex rec-02.tex
	evince 	 rec-02.pdf
	rm 		 rec-02.aux
	rm 		 rec-02.log
	rm 		 rec-02.out

rec-03:
	pdflatex rec-03.tex
	evince 	 rec-03.pdf
	rm 		 rec-03.aux
	rm 		 rec-03.log
	rm 		 rec-03.out



twelve-steps:
	pdflatex twelve-steps.tex
	evince 	 twelve-steps.pdf
	rm 		 twelve-steps.aux
	rm 		 twelve-steps.log
	rm 		 twelve-steps.out


