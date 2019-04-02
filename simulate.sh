#! /bin/bash

# define loop number
loop=1000
i=1

while(($i<=$loop))
do
    # clean all py process
    ps -ef | grep python | cut -c 9-15| xargs kill -s 9

    mkdir ~/cr/$i
    nohup python ~/PycharmProjects/cr/simulation.py > ~/cr/$i/rl_loss 2>&1 &

    printf "$i program in progress"

    time_start=$(date +%Y%m%d%s)

    time_now=$(date +%Y%m%d%s)

    time_temp=$(($time_now - $time_start))
    while(( $time_temp<=18000 ))
    do
        time_now=$(date +%Y%m%d%s)
        time_temp=$(($time_now - $time_start))
        printf "$time_temp"
        sleep 100
    done

    printf "finish"

    mv ~/cr/rl_model* ~/cr/$1
    let i=i+1
done

ps -ef | grep python | cut -c 9-15| xargs kill -s 9
