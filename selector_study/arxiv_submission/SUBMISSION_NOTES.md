# arXiv submission, SPEED DIAL selector study

Upload `arxiv_selector_study.tar.gz`. arXiv unpacks it, compiles `selector.tex` with
pdflatex, and publishes everything under `anc/` as ancillary files linked from the abstract
page.

**Categories.** Primary `math.NA` (Numerical Analysis). `cs.NA` is a formal alias, so the
paper lands in both listings and you do not choose between them. Cross-list `cs.LG` and
`cs.MS`.

**MSC 2020:** 65F08, 65F10, 65Y20, 68T05
**ACM class:** G.1.3; G.4; I.2.6

**Before you submit**

- Submit from Tricha's account. She is first author and the likely endorser for `math.NA`;
  a first submission to that category needs an endorsement.
- Choose the licence deliberately. arXiv defaults to non-exclusive-distribution, which is
  more restrictive than most people expect. CC BY 4.0 if the study should be reusable.
- The title, authors and abstract are in `selector.tex`; paste the abstract into the web
  form as plain text, with the LaTeX maths stripped.
- No figures, no .bbl, no BibTeX run. The bibliography is a `thebibliography` block inside
  the source, so a single pdflatex pass is enough and arXiv's two passes are plenty.

**What is deliberately not here.** No learned or discovered solver, no ICSI or AutoSpec
material, no comparison against the state of the art in preconditioning. This paper is
about evaluation methodology and says so in its limitations section.
