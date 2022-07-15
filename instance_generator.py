import os
import shutil
import csv
import numpy as np

def ri(low,high,m_one=0):
    p = np.random.uniform(0,1)
    if(m_one < p):
        return np.random.randint(low,high)
    return -1

distances = [[0,64100,64700,59000,68800,198000,190300,53600,54000,70000,189700,174900,68200,64700],
             [64100,0,10900,6200,6800,135700,131800,14800,10400,5900,131300,116400,13500,10900],
             [64700,10900,0,5800,4100,135700,125600,15100,11000,6000,125000,110200,17300,0],
             [59000,6200,5800,0,9800,141400,131300,9300,5300,11700,130800,116000,11600,5800],
             [68800,6800,4100,9800,0,131600,125000,19100,15100,1900,124500,109600,20300,4100],
             [198000,135700,135700,141400,131600,0,233000,144400,146100,129800,232500,217700,149200,135700],
             [190300,131800,125600,131300,125000,233000,0,140600,136600,126700,2500,26200,142900,125600],
             [53600,14800,15100,9300,19100,144400,140600,0,4400,20700,140100,125300,16600,15100],
             [54000,10400,11000,5300,15100,146100,136600,4400,0,16300,136000,121200,15300,11000],
             [70000,5900,6000,11700,1900,129800,126700,20700,16300,0,126100,111300,19400,6000],
             [189700,131300,125000,130800,124500,232500,2500,140100,136000,126100,0,25600,142300,125000],
             [174900,116400,110200,116000,109600,217700,26200,125300,121200,111300,25600,0,127500,110200],
             [68200,13500,17300,11600,20300,149200,142900,16600,15300,19400,142300,127500,0,17300],
             [64700,10900,0,5800,4100,135700,125600,15100,11000,6000,125000,110200,17300,0]]

speed                  = 333
coupling_time          = 15
decoupling_time        = 15
body_capacity          = 50
max_bodies             = 3
unscheduled_cost_coeff = 10000
service_time           = 2

def generate_instance(n_nodes,n_vessels,n_bodies,n_requests):
    directory = "random_instance"
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)    

    Nodes   = []
    Vessels = []
    Bodies  = []
    Requests = []
    
    with open(directory + "/nodes.csv","w") as f:
        writer = csv.writer(f)
        writer.writerow(["Node","is_client"])
        for i in range(n_nodes):
            is_client = -2 if ri(0,1,.2) < 0 else 1
            Nodes.append([i,is_client])
        writer.writerows(Nodes)
    
    with open(directory + "/vessels.csv","w") as f:
        writer = csv.writer(f)
        writer.writerow(["Vessel","start_location"])
        for i in range(n_vessels):
            Vessels.append([i,ri(0,n_nodes)])
        writer.writerows(Vessels)
    
    with open(directory + "/bodies.csv","w") as f:
        writer = csv.writer(f)
        writer.writerow(["Body","start_location"])
        for i in range(n_bodies):
            vessel = ri(0,n_vessels)
            Bodies.append([i,Vessels[vessel][1]])
        writer.writerows(Bodies)
    
    with open(directory + "/requests.csv","w") as f:
        writer = csv.writer(f)
        writer.writerow(["Request","pickup_location","pickup_from","pickup_to","delivery_location","delivery_from","delivery_to","num_containers","at_body"])
        for i in range(n_requests):
            at_body_id = ri(0,n_bodies,.7)
            pick_id    = -1 if at_body_id != -1 else ri(0,n_nodes)
            while(pick_id != -1 and Nodes[pick_id][1] == -2):
                pick_id = ri(0,n_nodes)
            pick_from  = -1 if at_body_id != -1 else ri(0,21600)
            pick_to    = -1 if at_body_id != -1 else pick_from + ri(0,5760) 
            deliv_id   = ri(0,n_nodes)
            while(deliv_id == pick_id or Nodes[deliv_id][1] == -2):
                deliv_id = ri(0,n_nodes)
            deliv_from = ri(0,21600) if at_body_id != -1 else pick_to + ri(7200,14400)
            deliv_to   = deliv_from + ri(0,5760)
            num_containters = ri(1,body_capacity/2)
            Requests.append((i,pick_id,pick_from,pick_to,deliv_id,deliv_from,deliv_to,num_containters,at_body_id))
        writer.writerows(Requests)
    
    with open(directory + "/distances.csv","w") as f:
        writer = csv.writer(f)
        writer.writerow([i for i in range(n_nodes)])
        writer.writerows([[distances[i][j] for j in range(n_nodes)] for i in range(n_nodes)])
    
    with open(directory + "/params.csv","w") as f:
        writer = csv.writer(f)
        writer.writerow(["speed","coupling_time","decoupling_time","body_capacity","max_bodies","total_number_of_nodes","total_number_of_vessels","total_number_of_bodies","total_number_of_requests","unscheduled_cost_coefficient","service_time"])
        data = [speed,coupling_time,decoupling_time,body_capacity,max_bodies,n_nodes,n_vessels,n_bodies,n_requests,unscheduled_cost_coeff,service_time]
        writer.writerow(data)
