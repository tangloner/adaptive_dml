#!/bin/bash
# $1 is the number of PSs
# $2 is the number of workers
# $3 is the result directory prefix

get_ps_conf(){
    ps=""
    for(( i=35; i > 35-$1; i-- ))
    do
        ps="$ps,ssd$i:2222"
    done
        ps=${ps:1}
};

get_worker_conf(){
    worker=""
    for(( i=35-$1; i > 35-$1-$2; i-- ))
    do
        worker="$worker,ssd$i:2222"
    done
    worker=${worker:1}
};

echo "release occupied ports in cluster"
./kill_cluster_pid.sh 26 35 2222

get_ps_conf $1
echo $ps
get_worker_conf $1 $2
echo $worker

mkdir ./result/$1"-"$2"-"$3
 
for(( i=35; i>35-$1-$2; i-- ))
do
{
    if [ $i == 35 ]
    then
        source /root/anaconda2/envs/tensorflow/bin/activate
        python /root/adaptive_dml/dnn_2.py $ps $worker --job_name=ps --task_index=0 >> /root/adaptive_dml/result/$1"-"$2"-"$3/ps0.txt
        echo $ps $worker " done."
    else
	ssh ssd$i "source activate tensorflow"
        n=`expr 35 - $1`
        if [ $i -gt $n ]
        then
            index=`expr 35 - $i`
            ssh ssd$i python /root/adaptive_dml/dnn_2.py $ps $worker --job_name=ps --task_index=$index >> /root/adaptive_dml/result/$1"-"$2"-"$3/ps"$index".txt
            echo "ps "$index "done!"
        else
            index=`expr 35 - $1 - $i`
            ssh ssd$i python /root/adaptive_dml/dnn_2.py $ps $worker --job_name=worker --task_index=$index >> /root/adaptive_dml/result/$1"-"$2"-"$3/worker"$index".txt
            echo "worker "$index " done!" 
	fi
    fi
}&
done 
wait
echo "done"



