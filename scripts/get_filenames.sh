#!/bin/bash
for file in /snk/Data/img_align_celeba/*
do
    if test -f $file
    then
        readlink -f $file > filename.txt
    fi
done
