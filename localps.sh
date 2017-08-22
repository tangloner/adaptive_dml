#!/bin/bash
# $1 is the number of PSs
# $2 is the number of workers
# ps.sh run in ssd35

get_ps_conf(){
    ps=""
    for(( i=2222; i > 2222-$1; i-- ))
    do
        ps="$ps,localhost:$i"
    done
        ps=${ps:1}
};

get_worker_conf(){
    worker=""
    for(( i=2222-$1; i > 2222-$1-$2; i-- ))
    do
        worker="$worker,localhost:$i"
    done
    worker=${worker:1}
};


get_ps_conf $1
echo $ps
get_worker_conf $1 $2
echo $worker

mkdir ./result/$1"-"$2
 
for(( i=2222; i>2222-$1-$2; i-- ))
do
{
	n=`expr 2222 - $1`
    if [ $i -gt $n ]
	then
    	index=`expr 2222 - $i`
        python ./example.py $ps $worker --job_name=ps --task_index=$index > ./result/$1"-"$2/ps"$index".txt
        echo "ps: "$index "done!"
	else
    	index=`expr 2222 - $1 - $i`
        python ./example.py $ps $worker --job_name=worker --task_index=$index > ./result/$1"-"$2/worker"$index".txt
        echo "worker: "$index "done!"
	fi
}&
done 
wait
echo "done"



