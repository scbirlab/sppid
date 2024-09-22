#!/usr/bin/env bash

set -e
set -x

INPUT1=test/inputs/DYR_YEAST.a3m
INPUT2=test/inputs/CAPZA_YEAST.a3m
# yunta dca-single $INPUT1 -2 $INPUT2 -o test/outputs/dca-single.tsv --plot test/outputs/dca-single --apc \
# && touch test/outputs/dca-single.success

INPUTA=test/inputs/DYR_YEAST.a3m
INPUTB=(test/inputs/CAPZA_YEAST.a3m test/inputs/WWM1_YEAST.a3m)
# yunta dca-many $INPUTA -2 "${INPUTB[@]}" --apc -o test/outputs/dca-many.tsv --plot test/outputs/dca-many \
# && touch test/outputs/dca-many.success

FILE1=test/outputs/file1.txt
FILE2=test/outputs/file2.txt
echo $INPUTA > $FILE1
echo "${INPUTB[@]}" | tr ' ' $'\n' > $FILE2
# yunta dca-single $FILE1 -2 $FILE2 --apc -o test/outputs/dca-many-list.tsv --plot test/outputs/dca-many-list --list-file \
# && touch test/outputs/dca-many-list.success

if [ $(diff test/outputs/dca-many-list.tsv test/outputs/dca-many.tsv | wc -l) != 0 ]
then 
    >&2 echo "ERROR: List and basic inputs gave different outputs!"
    exit 1
fi

if [ ! -e "test/outputs/dca-many.success" ]
then 
    >&2 echo "ERROR: DCA did not succeed!"
    exit 1
fi

if [ -z $1 ]
then
    # yunta rf2t-single $INPUT1 -2 $INPUT2 -o test/outputs/rf2t-single.tsv --plot test/outputs/rf2t-single \
    # && touch test/outputs/rf2t-single.success
    # yunta rf2t-single $INPUTA -2 "${INPUTB[@]}" -o test/outputs/rf2t-many.tsv --plot test/outputs/rf2t-many \
    # && touch test/outputs/rf2t-many.success
    # yunta rf2t-single $FILE1 -2 $FILE2 -o test/outputs/rf2t-many-list.tsv --plot test/outputs/rf2t-many-list --list-file \
    # && touch test/outputs/rf2t-many-list.success
    # if [ $(diff test/outputs/rf2t-many-list.tsv test/outputs/rf2t-many.tsv | wc -l) != 0 ]
    # then 
    #     >&2 echo "ERROR: List and basic inputs gave different outputs!"
    #     exit 1
    # fi

    yunta af2-single $INPUTA -2 "${INPUTB[@]}" -o test/outputs/af2-many \
    && touch test/outputs/af2-many.success
    yunta af2-single $FILE1 -2 $FILE2 -o test/outputs/af2-many-list --list-file \
    && touch test/outputs/af2-many-list.success
fi
