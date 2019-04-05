#!/bin/bash

# mkdir -p /results/
python script_run-stopword-statistics.py
python script_make-figure-01.py
python script_make-figure-02a.py
python script_make-figure-02b.py
python script_make-figure-02c.py
python script_make-figure-02d.py
