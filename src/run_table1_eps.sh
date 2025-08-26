#!/bin/bash

if [ "$1" != "qing" ] && [ "$1" != "ruo" ]; then
    echo "Usage: $0 {qing|ruo}"
    exit 1
fi

COMPUTER=$1

javac UsacoProbs/Table1Synthetic.java UsacoProbs/clique2_mk_benchmark_accuracy.java

if [ "$COMPUTER" = "qing" ]; then
    echo "Running qing tasks (first half)..."
    java UsacoProbs.Table1Synthetic 1e-6
    java UsacoProbs.Table1Synthetic 1e-5
    java UsacoProbs.Table1Synthetic 1e-4
    java UsacoProbs.Table1Synthetic 1e-3
    java UsacoProbs.Table1Synthetic 1e-2
    java UsacoProbs.Table1Synthetic 0.1
    java UsacoProbs.Table1Synthetic 1
else
    echo "Running ruo tasks (second half)..."
    java UsacoProbs.Table1Synthetic 1e2
    java UsacoProbs.Table1Synthetic 1e3
    java UsacoProbs.Table1Synthetic 1e4
    java UsacoProbs.Table1Synthetic 1e5
    java UsacoProbs.Table1Synthetic 1e6
fi
