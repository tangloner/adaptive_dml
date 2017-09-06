#!/bin/sh
#$1 is the number of PSs
#$2 is the number of machines
#$3 is the No. of first machines eg:ssd32's "No." is 32

#global veriable
cluster=""
py_command=""
file_index=1
pid=""

#get the pid of target port
get_pid(){
#$1 ip
#$2 target port
re=`ssh $1 "netstat -anp|grep "$2`
tem=${re%%/*}
pid=${tem##* }
if [ "$pid" == "" ]
then
    echo "the port:"$2" of "$1" is free"
else
    echo "the port:"$2" of "$1" has been occpuied by process："$pid
fi
}

#get the cluster confige of tensorflow
get_cluster_conf(){
    ps=""
    worker=""
    n=0
    for(( i=0; i<$2; i++ ));
    do
        index=`expr $i + $3`
        if [ $i -lt $1 ]
        then
            ps="$ps,172.17.10.$index:2222"
        else
            worker="$worker,172.17.10.$index:2222"
        fi
    done
    cluster="--ps_hosts="${ps:1}" --worker_hosts="${worker:1}

};

#get the tensorflow command
 get_py_command(){
    #$1:number of cycle
    #$2:numbkkjer of PSs
    #$3:number of workers
    #$4:n_rest
    #$5:n_batch
    #$6:cluster_conf
    if [ $1 -lt $2 ]
    then
        py_command=" python /root/code/LogicRgression/offline_logistic_regression.py  --job_name=ps --task_index="$1" "$cluster
    else
        index=`expr $1 - $2`
    echo "$index"
        start_batch_index=$file_index
    end_batch_index=`expr $file_index  + $5`
        if [ $index -lt $4 ]
        then
        echo "$4"
            end_batch_index=$[end_batch_index+1]
        fi
    file_index=$end_batch_index
        py_command="python /root/code/LogicRgression/offline_logistic_regression.py  --job_name=worker --task_index=${index} --batch_size=500 "$cluster" --num_epochs=5 --train=/root/data/url_svmlight/Day#.svm  --#line_skip_count=0 --features=3231961 --data_index_start=${start_batch_index} --data_index_end=${end_batch_index}"
    fi
}

get_cluster_conf $1 $2
n_batch=`expr 120 / $2`
n_rest=`expr 120 % $2`

for(( step=0; step<$2; step++ ));
do
    
    ip=`expr $step + $3`
    num_workers=`expr $2 + $1`
    get_py_command $step $1 $num_workers $n_rest $n_batch
{
    get_pid "ssd"$ip 2222
    #ssh "ssd"$ip "kill "$pid
    pid=""
    count=0
    #if pid=="", restart this cmd for 3 times
    #until [ "$pid" != "" -o $count -lt 3 ]
    #do
    cout=`expr $count + 1`
    	#ssh "ssd"$ip "source activate tensorflow"
        #ssh "ssd"$ip "$py_command" >> ./result/"ssd"$ip .txt
    echo "$py_command"
    get_pid "ssd"$ip  2222
    #done
}&
done
wait
echo "Game Over"
