#!/bin/bash

rm -r demos
cp -r ../code/demos source/

sphinx-build -M html source .
make html

rm -r _sources
rm -r _static

mv html/* .

rm -r html