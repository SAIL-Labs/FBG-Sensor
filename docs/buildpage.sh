#!/bin/bash

sphinx-build -M html source .
make html

rm -r _sources
rm -r _static

mv html/* .

rm -r html