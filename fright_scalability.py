from vessel_swap_body_class import vsbp
from instance_generator import generate_instance
import csv
import numpy as np
from datetime import datetime
import os

with open("times.txt","w") as f:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    f.write("Started at " + current_time)
    f.close()

def freight():
    with open("fright_scalability.csv","w") as f:
        w = csv.writer(f)
        w.writerow(["n_nodes","n_vessels","n_bodies","n_requests","avg_time","std_dev"])
        instances = []
        n_nodes   = 2
        n_vessels = 2
        n_bodies  = 2
        n_requests = 2
        for i in range(3):
            instances.append([n_nodes,n_vessels,n_bodies,n_requests])
            n_nodes    += 2
            n_vessels  += 2
            n_bodies   += 2
        for i in range(3):
            times = []
            for _ in range(3):
                generate_instance(*instances[i])
                test = vsbp("random_instance")
                test.solve_vsbr(gap=0.2,time_limit = 1200)
                times.append(test.opt_mod.Runtime)
            avg = np.mean(times)
            std = np.std(times,ddof = 1)
            w.writerow(instances[i] + [avg,std])

def requests():
    with open("request_scalability.csv","w") as f:
        w = csv.writer(f)
        w.writerow(["n_nodes","n_vessels","n_bodies","n_requests","avg_time","std_dev"])
        instances = []
        n_nodes   = 2
        n_vessels = 2
        n_bodies  = 2
        n_requests = 1
        for i in range(3):
            instances.append([n_nodes,n_vessels,n_bodies,n_requests])
            #n_nodes    += 2
            #n_vessels  += 1 if i < 1 else 0
            #n_bodies   += 1 if i < 2 else 0
            #n_requests += 1
        for i in range(3):
            times = []
            for _ in range(5):
                generate_instance(*instances[i])
                test = vsbp("random_instance")
                test.solve_vsbr(gap=0.2,time_limit = 1200)
                times.append(test.opt_mod.Runtime)
            avg = np.mean(times)
            std = np.std(times,ddof = 1)
            w.writerow(instances[i] + [avg,std])

requests()

with open("times.txt","a") as f:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    f.write("Finished at " + current_time)
    f.close()
#os.system('shutdown -s')
