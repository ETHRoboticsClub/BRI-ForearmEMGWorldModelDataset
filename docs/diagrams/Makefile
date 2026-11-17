# Render every diagram to SVG: `make -C docs/diagrams` (needs the typst Python package: pip install -e ".[viz]")
# Only diagrams whose .typ (or style.typ) changed are rendered again.

PYTHON ?= python
SVGS := $(patsubst %.typ,%.svg,$(filter-out style.typ,$(wildcard *.typ)))

all: $(SVGS)

%.svg: %.typ style.typ
	$(PYTHON) -c "import sys, typst; open(sys.argv[2], 'wb').write(typst.compile(sys.argv[1], format='svg') + b'\\n')" $< $@

.PHONY: all
