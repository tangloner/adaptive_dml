#!/bin/bash
# $1 is the number of PSs
# $2 is the number of workers
# ps.sh run in ssd42

get_ps_conf(){
    ps=""
    for(( i=42; i > 42-$1; i-- ))
    do
        ps="$ps,ssd$i:2222"
    done
        ps=${ps:1}
};

get_worker_conf(){
    worker=""
    for(( i=42-$1; i > 42-$1-$2; i-- ))
    do
        worker="$worker,ssd$i:2222"
    done
    worker=${worker:1}
};

for(( i=0; i<$2; i++ ))
do
{
    echo "0">temp$i
}
done

get_ps_conf $1
echo $ps
get_worker_conf $1 $2
echo $worker


mkdir ./result/$1"-"$2".async"
 
for(( i=42; i>42-$1-$2; i-- ))
do
{
    if [ $i == 42 ]
    then
        source /root/anaconda2/envs/tensorflow/bin/activate
        python /root/adaptive_dml/dnn_1.py $ps $worker --job_name=ps --task_index=0 >> /root/adaptive_dml/result/$1"-"$2".async"/ps0.txt
        echo $ps $worker " done."
    else
	ssh ssd$i "source activate tensorflow"
        n=`expr 42 - $1`
        if [ $i -gt $n ]
        then
            index=`expr 42 - $i`
            ssh ssd$i python /root/adaptive_dml/dnn_1.py $ps $worker --job_name=ps --task_index=$index >> /root/adaptive_dml/result/$1"-"$2".async"/ps"$index".txt
            echo "ps "$index "done!"
        else
            index=`expr 42 - $1 - $i`
            ssh ssd$i python /root/adaptive_dml/dnn_1.py $ps $worker --job_name=worker --task_index=$index >> /root/adaptive_dml/result/$1"-"$2".async"/worker"$index".txt
            echo "worker "$index " done!" 
            echo "1">temp$index
	fi
    fi
}&
done 
wait
echo "done"

while true
do
    flag=0
    for(( i=0; i<$2; i++ ))
    do
    {   
	tem=`cat temp$i`
	flag=`expr $tem + $flag`
    }
    done	
    if [ $flag == $2 ]
    then
    	./kill_cluster_pid.sh 26 42 2222
	break
    fi
done 
rm -f temp*
echo "work done"


