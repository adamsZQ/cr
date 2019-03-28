#! /bin/bash

# define loop number
loop=1
i=1
cpu=8
cpu_num=1
while(($i<=$loop))
do
    printf "%dth loop" $i

    while(($cpu_num<=cpu))
    do
         printf "%dth cpu" $cpu_num
    done
done
