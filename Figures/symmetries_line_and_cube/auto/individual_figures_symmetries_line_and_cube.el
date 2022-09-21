(TeX-add-style-hook
 "individual_figures_symmetries_line_and_cube"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("standalone" "tikz" "crop" "convert={density=200,outext=.png}" "border=0.4cm")))
   (TeX-run-style-hooks
    "latex2e"
    "./Input/line"
    "./Input/cube"
    "standalone"
    "standalone10"
    "pgfplots"
    "amsmath"
    "physics"
    "xcolor")
   (LaTeX-add-xcolor-definecolors
    "lin_1"
    "lin_2"
    "lin_3"
    "cube_1"
    "cube_2"
    "cube_3"))
 :latex)

