
globals = ["10"]#["5", "10"]
global_metrics = ["loss", "accuracy"]
locals = ["5"]#["5", "10"]
rounds = ["10"]#["5", "10"]
averages = ["simple", "std_dev", "weighted_metric"]
metrics = ["loss", "accuracy"]

exp_no = 0

'''
for round in rounds:
    for global_epochs in locals:
        for average in averages:
            for metric in metrics:
                
                print( "CUDA_VISIBLE_DEVICES=0 python3 run_clients.py covid_xray jsrt.mat CovidMix_fd_client_zeus.py 5 " +round+" "+global_epochs+" .6 .1 10 1 0 "+metric+" "+average+'&>> log_global_users_5_rounds_' +round+'_'+'local_epochs_' +global_epochs+'_'+metric+'_'+average+'.txt &')
                
                
                exp_no+=1
                
                print()
                
'''

for global_epochs in globals:
    for round in rounds:
        for local_epochs in locals:
            for average in averages:
                for metric in metrics:
                
                
                
                    print( "CUDA_VISIBLE_DEVICES=0 python3 run_clients_transfer_fl.py covid_xray_transfer jsrt.mat CovidMix_fd_client_zeus_transfer_fl.py 5 " +round+" "+local_epochs+" .6 .1 10 1 0 "+metric+" "+average+ " exp_transfer_dir_global_epochs_"+global_epochs+"_loss/best_model_0.pth global_epoch_" +global_epochs+'&>> log_transfer_fl_users_5_rounds_' +round+'_'+'global_epochs_' +global_epochs+'_local_epochs_' +local_epochs+'_'+metric+'_'+average+'.txt &')
                
                    
                    exp_no+=1
                
                    print()

print(exp_no)
                
                
                
                
                
                
                
        




