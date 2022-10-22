#!/usr/bin/bash

Mbytes=$((256*5));
echo "Generating big mattr";
#./generator.o "$Mbytes" 64 input.txt;

for i in {1..5};
do
    Mbytes=$((256*$i));
    M=$((1048576*$i));
    echo "$Mbytes MB";
    echo "M=$M";
    ./2.o "$M" 64 input.txt output.txt;
done;
