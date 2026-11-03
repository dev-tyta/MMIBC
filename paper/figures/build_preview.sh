#!/usr/bin/env bash
# usage: build_preview.sh <fragment-name-without-.tex> [outdir]   -> <outdir>/<name>.pdf
set -euo pipefail
name="$1"; out="${2:-build}"; here="$(cd "$(dirname "$0")" && pwd)"; mkdir -p "$out"
cat > "$out/$name.wrap.tex" <<TEX
\documentclass[border=3pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\input{$here/mmibc_tikz}
\begin{document}
\input{$here/$name}
\end{document}
TEX
( cd "$out" && pdflatex -interaction=nonstopmode -halt-on-error "$name.wrap.tex" >"$name.log" 2>&1 ) || { tail -30 "$out/$name.log"; exit 1; }
mv "$out/$name.wrap.pdf" "$out/$name.pdf"
