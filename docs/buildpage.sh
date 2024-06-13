#!/bin/bash

rm -r source/demos
cp -r ../code/demos source/

sphinx-build -M html source .
make html

./refresh.sh

mv html/* .

rm -r html