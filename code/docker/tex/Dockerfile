# LaTeX for compile-testing the report and its generated fragments locally:
# pdfLaTeX with -shell-escape (minted needs pygmentize), biber/biblatex,
# pgfplots and the packages rho.cls loads. The submitted report is compiled
# in Overleaf; this only catches errors early.
FROM debian:bookworm-20241016-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends texlive-latex-base texlive-latex-recommended \
        texlive-latex-extra texlive-pictures texlive-science texlive-fonts-recommended \
        texlive-fonts-extra texlive-bibtex-extra biber python3-pygments \
 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 dev
USER dev
WORKDIR /work
