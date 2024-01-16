#! /bin/bash

CLIENTS=("10.200.61.42" "10.200.62.229" "10.200.19.136" "10.200.61.116" "10.200.3.81" "10.200.40.115" "10.200.16.60" "10.200.40.196" "10.200.28.142")
USERNAME="mahbub"
PASSWORD="xxx"

SCRIPT="echo -e 'PASSWORD' | sudo -S reboot -h now"

for i in ${!CLIENTS[*]} ; do

     echo "Client: ${i}, Address: ${CLIENTS[i]}"
     
     SCR=${SCRIPT/PASSWORD/$PASSWORD}
     
     sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@${CLIENTS[i]} "${SCR}"

done


