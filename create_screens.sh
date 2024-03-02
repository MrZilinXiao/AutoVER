#!/bin/bash
if [ $# -lt 2 ]
then
echo "Usage: ${prog##*/} sessName cmd1 cmd2 ..."
exit -1
fi

sessName=$1
shift

screen -X -S $sessName quit > /dev/null 2>&1

screen -dmS $sessName

until [ $# -eq 0 ]
do
    cmd=$1
    echo "exec cmd: $cmd"
    screen -S $sessName -p 0 -X stuff "$cmd"$'\n'
    shift;
done