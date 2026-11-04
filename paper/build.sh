#!/bin/sh
# Build paper.pdf (via Chrome) and paper.docx from paper.md. Needs pandoc and Google Chrome.
cd "$(dirname "$0")"
pandoc paper.md -f markdown+subscript -t html5 --template template.html -o paper.html
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu \
  --no-pdf-header-footer --print-to-pdf=paper.pdf "file://$PWD/paper.html"
pandoc paper.md -f markdown+subscript -o paper.docx
