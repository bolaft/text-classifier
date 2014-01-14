#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
:Name:
    benchmark.py

:Authors:
    Soufian Salim (soufi@nsal.im)

:Date:
    14 january 2014 (creation)

:Description:
    Benchmark routine for NLTK text classifiers
"""

import sys
import subprocess

CLASSIFY_FILE = "classify.py"


# Main
def main(argv):
    if len(argv) > 0:
        sys.exit("This script takes no argument")

    for c in ["NB", "SVM", "DT", "KNN"]:
        for f in ["tfidf", "chi2"]:
            print("Running " + CLASSIFY_FILE + " with parameters \"" + c + "\" and \"" + f + "\"")
            subprocess.call("python " + CLASSIFY_FILE + " -c " + c + " -f " + f, shell=True)

# Launch
if __name__ == "__main__":
    main(sys.argv[1:])
