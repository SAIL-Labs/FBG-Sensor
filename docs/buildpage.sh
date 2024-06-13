#!/bin/bash

rm -r source/demos
cp -r ../code/demos source/

sphinx-build -M html source .
make html

rm -r _sources
rm -r _static
rm -r _images

mv html/* .

rm -r html