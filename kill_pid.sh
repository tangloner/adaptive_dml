#!/bin/bash
#$1:the start index of cluster eg；ssd32-32
#$2:the end index of cluster 
#$3:target port

#global variable
pid=""

for((step=$1;step>$1-$2;step--));
do
	re=`netstat -anp|grep $step`
	echo "kill process at " $step
	tem=${re%%/*}
	pid=${tem##* }
	if [ "$pid" != "" ]
	then
	    kill $pid
	fi
done
echo "clean up"
