#!/bin/bash
echo "Hello World"

#declare -a array=("element1" "element2" "element3")
declare -a models=("FGNODE" "GNODE" "HGN" "HGNN" "LGNN")

declare -a datapoints=("100" "500" "1000" "5000")
for i in "${models[@]}"
do
	for j in "${datapoints[@]}"
    do
        python "Pendulum-C$i-post.py" $j &
    done
done

for i in "${models[@]}"
do
	for j in "${datapoints[@]}"
    do
        python "Pendulum-$i-post.py" $j &
    done
done

for i in "${models[@]}"
do
	for j in "${datapoints[@]}"
    do
        python "Spring-$i-post.py" $j &
    done
done
wait

