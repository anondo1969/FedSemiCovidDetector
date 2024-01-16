#! /bin/bash

CLIENTS=("10.200.61.42" "10.200.62.229" "10.200.19.136" "10.200.61.116" "10.200.3.81" "10.200.40.115" "10.200.16.60" "10.200.40.196" "10.200.28.142")


USERNAME="mahbub"
PASSWORD="29061969"

file_name=$1

SCRIPT="cd Desktop/0_my_code; rm ${file_name}*; exit"
     
SCR=${SCRIPT/PASSWORD/$PASSWORD}
     


for client in ${!CLIENTS[*]} ; do

     echo "Client: ${client}, Address: ${CLIENTS[client]}"
     
     sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@${CLIENTS[client]} "${SCR}"
     

done

