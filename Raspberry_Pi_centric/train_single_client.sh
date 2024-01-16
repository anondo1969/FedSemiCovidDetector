#! /bin/bash


CLIENTS=("10.200.61.42" "10.200.62.229" "10.200.19.136" "10.200.61.116" "10.200.3.81" "10.200.40.115" "10.200.16.60" "10.200.40.196" "10.200.28.142")


USERNAME="mahbub"
PASSWORD="xxx"



#users = int(sys.argv[1])
#client_order = int(sys.argv[2])
#port = int(sys.argv[3])
#host = str(sys.argv[4])
#all_data_use = bool(sys.argv[5])
#classification_batch_size = int(sys.argv[6])
#segmentation_batch_size = int(sys.argv[7])
#classification_label_portion = float(sys.argv[8])
#segmentation_label_portion = float(sys.argv[9])

host=$1
port=$2
users=$3
all_data_use=$4
classification_batch_size=$5
segmentation_batch_size=$6
classification_label_portion=$7
segmentation_label_portion=$8
client=$9
log_file=${10}

echo ""
echo "Server or host address: ${host}"
echo "Server port: ${port}"
echo "Total clients: ${users}"
echo "Use all data?: ${all_data_use}"
echo "Classification batch size: ${classification_batch_size}"
echo "Segmentation batch size: ${segmentation_batch_size}"
echo "Classification label portion: ${classification_label_portion}"
echo "Segmentation label portion: ${segmentation_label_portion}"
echo "log_file: ${log_file}"
echo ""




echo "Client: ${client}, Address: ${CLIENTS[client]}"
     
SCRIPT="cd Desktop/0_my_code; export OPENBLAS_NUM_THREADS=1; python3 CovidMix_fd_client.py ${users} ${client} ${port} ${host} ${all_data_use} ${classification_batch_size} ${segmentation_batch_size} ${classification_label_portion} ${segmentation_label_portion}&>> {log_file}_${client} & exit"
     
SCR=${SCRIPT/PASSWORD/$PASSWORD}
     
sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@${CLIENTS[client]} "${SCR}"
     
#sleep 30
     

echo ""


#bash ./train_single_client.sh 10.200.49.191 1965 3 1 10 1 0.6 0.1 0 log_clients_3_test

