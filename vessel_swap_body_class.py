from gurobipy import *
from itertools import combinations
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib._color_data as mcd
import random

class vsbp:
    
    def __init__(self, instance_folder = None, name = "vessel swap body routing problem"):
        self.instance_folder = instance_folder
        self._get_dimensions()
        self._get_instance()
        self.opt_mod = Model(name = name)    
    
    def _get_dimensions(self):
        self.Params     = pd.read_csv(self.instance_folder + "/params.csv")
        self.n_nodes    = self.Params["total_number_of_nodes"][0]
        self.n_vessels  = self.Params["total_number_of_vessels"][0]
        self.n_bodies   = self.Params["total_number_of_bodies"][0]
        self.n_requests = self.Params["total_number_of_requests"][0]

    def _get_instance(self):
        self.Nodes     = pd.read_csv(self.instance_folder + "/nodes.csv")
        self.Vessels   = pd.read_csv(self.instance_folder + "/vessels.csv")
        self.Bodies    = pd.read_csv(self.instance_folder + "/bodies.csv")
        self.Requests  = pd.read_csv(self.instance_folder + "/requests.csv")
        self.Params    = pd.read_csv(self.instance_folder + "/params.csv")
        self.Distances = pd.read_csv(self.instance_folder + "/distances.csv").to_numpy() 
    
    def solve_vsbr(self,gap=.1,plot=True):
        
        ########## Generating network nodes ##########################################

        self._generate_network_nodes()

        ########## Generating network arcs ##########################################

        self._generate_network_arcs()

        ########## Generating travel times ##########################################

        self._generate_travel_times()

        ########## Generating decision variables ####################################

        self._generate_decision_variables()

        ########## Generating objective function ####################################

        self.obj_fn = quicksum(quicksum(
                            self.travel_time[a]*self.z_v[v][a] 
                                for a in self.A) 
                                for v in range(self.n_vessels)) + quicksum(
                            self.Params["unscheduled_cost_coefficient"][0] * (1 - self.gamma_r[r]) 
                                for r in range(self.n_requests))
        self.opt_mod.setObjective(self.obj_fn,GRB.MINIMIZE)

        ########## Generating constraints ###########################################

        # Routing for vessels and bodies costraints
        self._generate_routing_constraints()

        # Sub-tour elimination for vessel routes
        self._generate_sub_tour_elimination_constraints()

        # Capacity constraints
        self._generate_capacity_constraints()

        # Serving requests
        self._generate_serving_requests_constraints()

        # Time constraints
        self._generate_time_constraints()

        # Transfers at non-client nodes
        self._generate_transfer_constraints()

        # Valid inequalities
        self._generate_valid_inequalities()

        ########## Setting parameters and solving the problem #########################

        self.opt_mod.update()
        self.opt_mod.Params.MIPGap=gap
        self.opt_mod.optimize()
        print(f'Objective Function Value: {self.opt_mod.objVal}')    
    
    def _generate_network_nodes(self):
        self.nodes_requests      = {}
        self.nodes_locations     = {}
        self.Pickup_nodes        = []
        self.Pickup_nodes_copy   = []
        self.Delivery_nodes      = []
        self.Delivery_nodes_copy = []

        for i in range(self.n_requests):
            if(self.Requests["pickup_location"][i] != -1):
                label = "P_" + str(i)
                self.Pickup_nodes.append(label)
                self.nodes_requests[label] = i
                self.nodes_locations[label] = self.Requests["pickup_location"][i]

                label_copy = "P'_" + str(i + self.n_requests)
                self.Pickup_nodes_copy.append(label_copy)
                self.nodes_requests[label_copy] = i
                self.nodes_locations[label_copy] = self.Requests["pickup_location"][i]

            if(self.Requests["delivery_location"][i] != -1):
                label = "D_" + str(i + 2*self.n_requests)
                self.Delivery_nodes.append(label)
                self.nodes_requests[label] = i
                self.nodes_locations[label] = self.Requests["delivery_location"][i]

                label_copy = "D'_" + str(i + 3*self.n_requests)
                self.Delivery_nodes_copy.append(label_copy)
                self.nodes_requests[label_copy] = i
                self.nodes_locations[label_copy] = self.Requests["delivery_location"][i]

        self.I = self.Pickup_nodes + self.Pickup_nodes_copy + self.Delivery_nodes + self.Delivery_nodes_copy

        self.S_b = [[] for _ in range(self.n_bodies)]
        for i in [loc for loc in range(self.n_nodes) if self.Nodes["is_client"][loc] == -2]:
            for b in range(self.n_bodies):
                for c in [1,2]:
                    label = "S_" + str(i) + "_" + str(b) + "_" + str(c)
                    self.S_b[b].append(label)
                    self.nodes_locations[label] = i
        self.S = [s for s_b in self.S_b for s in s_b]

        self.J_0 = []
        for i in range(self.n_vessels):
            label = "o_v_" + str(i)
            self.J_0.append(label)
            self.nodes_locations[label] = self.Vessels["start_location"][i]
        for i in range(self.n_bodies):
            label = "o_b_" + str(i)
            self.J_0.append(label)
            self.nodes_locations[label] = self.Bodies["start_location"][i]

        self.Dummy = ["sink"]
        self.nodes_locations["sink"] = -1

        self.N = self.J_0 + self.I + self.S + self.Dummy
    
    def _generate_network_arcs(self):
        self.A_1 = tuplelist()
        self.A_2 = tuplelist()
        self.A_3 = tuplelist()
        self.A_4 = tuplelist()
        self.A_5 = tuplelist()
        self.A_sink = tuplelist()
        self.A_D = tuplelist()
        self.A = tuplelist()

        # A_1
        for o in self.J_0 + self.S:
            for n in self.N:
                if(o == n):
                    continue
                self.A_1.append((o,n))
        ###################################################################
        # A_2
        for p,p_copy in zip(self.Pickup_nodes,self.Pickup_nodes_copy):
            self.A_2.append((p,p_copy))
        for p in self.Pickup_nodes:
            r = self.nodes_requests[p]
            for n in self.N:
                if(n[0] == 'P' or n[0] == 'D'):
                    if(self.nodes_requests[n] == r):
                        continue
                self.A_2.append((p,n))
        ###################################################################
        # A_3
        for p_copy,d in zip(self.Pickup_nodes_copy,self.Delivery_nodes):
            self.A_3.append((p_copy,d))
        for p_copy in self.Pickup_nodes_copy:
            r = self.nodes_requests[p_copy]
            for n in self.N:
                if(n[0] == 'P' or n[0] == 'D'):
                    if(self.nodes_requests[n] == r):
                        continue
                self.A_3.append((p_copy,n))
        ###################################################################
        # A_4
        for d,d_copy in zip(self.Delivery_nodes,self.Delivery_nodes_copy):
            self.A_4.append((d,d_copy))
        for d in self.Delivery_nodes:
            r = self.nodes_requests[d]
            for n in self.N:
                if(n[0] == 'P' or n[0] == 'D'):
                    if(self.nodes_requests[n] == r):
                        continue
                self.A_4.append((d,n))
        ###################################################################
        # A_5
        for d_copy in self.Delivery_nodes_copy:
            r = self.nodes_requests[d_copy]
            for n in self.N:
                if(n[0] == 'P' or n[0] == 'D'):
                    if(self.nodes_requests[n] == r):
                        continue
                self.A_5.append((d_copy,n))
        ###################################################################
        # A_sink
        for n in self.J_0 + self.I + self.S:
            self.A_sink.append((n,self.Dummy[0]))

        self.A = self.A_1 + self.A_2 + self.A_3 + self.A_4 + self.A_5 #+ self.A_sink

        # A_D
        for i,j in self.A:
            if(self.nodes_locations[i] == self.nodes_locations[j]):
                self.A_D.append((i,j))
        self.A_D += self.A_sink
    
    def _generate_travel_times(self):
        self.travel_time = {}
        for dep,arr in self.A:
            time = self.Distances[self.nodes_locations[dep]][self.nodes_locations[arr]] / self.Params["speed"][0] if self.nodes_locations[arr] != -1 else 0

            if(dep in self.I and arr in self.I and self.nodes_requests[dep] == self.nodes_requests[arr]):
                service_time = self.Params["coupling_time"][0] if self.A_1.select(dep,arr) else self.Params["decoupling_time"][0]
                r = self.nodes_requests[dep]
                time = service_time * self.Requests["num_containers"][r]

            self.travel_time[(dep,arr)] = time
        
    def _generate_decision_variables(self):
        self.z_v      = []
        self.beta_b   = []
        self.omega_bv = []
        self.x_br     = []
        self.delta_b  = []
        self.gamma_r  = []
        self.tau_a_v  = []
        self.tau_d_v  = []
        self.tau_a_b  = []
        self.tau_d_b  = []
        self.y_v      = []
        self.phi_b    = []

        for v in range(self.n_vessels):
            self.z_v.append({})
            for arc in self.A:
                self.z_v[v][arc] = self.opt_mod.addVar(name = "z_" + str(v) + "_(" + arc[0] + "," + arc[1] + ")", vtype = GRB.BINARY)

        for b in range(self.n_bodies):
            self.beta_b.append({})
            for arc in self.A:
                self.beta_b[b][arc] = self.opt_mod.addVar(name = "beta_" + str(b) + "_(" + arc[0] + "," + arc[1] + ")", vtype = GRB.BINARY)

        for v in range(self.n_vessels):
            self.omega_bv.append([])
            for b in range(self.n_bodies):
                self.omega_bv[v].append({})
                for arc in self.A:
                    self.omega_bv[v][b][arc] = self.opt_mod.addVar(name = "omega_" + str(b) + "_" + str(v) + "_(" + arc[0] + "," + arc[1] + ")", vtype = GRB.BINARY)

        for r in range(self.n_requests):
            self.x_br.append([])
            for b in range(self.n_bodies):
                self.x_br[r].append({})
                for arc in self.A:
                    self.x_br[r][b][arc] = self.opt_mod.addVar(name = "x_" + str(b) + "_" + str(r) + "_(" + arc[0] + "," + arc[1] + ")", vtype = GRB.BINARY)

        for b in range(self.n_bodies):
            self.delta_b.append({})
            for arc in self.A:
                self.delta_b[b][arc] = self.opt_mod.addVar(name = "delta_" + str(b) + "_(" + arc[0] + "," + arc[1] + ")", vtype = GRB.BINARY)

        for r in range(self.n_requests):
            self.gamma_r.append(self.opt_mod.addVar(name = "gamma_" + str(r), vtype = GRB.BINARY))

        for v in range(self.n_vessels):
            self.tau_a_v.append({})
            for n in self.N:
                self.tau_a_v[v][n] = self.opt_mod.addVar(name = "tau_a_" + str(v) + "_" + n, vtype = GRB.CONTINUOUS, lb = 0)

        for v in range(self.n_vessels):
            self.tau_d_v.append({})
            for n in self.N:
                self.tau_d_v[v][n] = self.opt_mod.addVar(name = "tau_d_" + str(v) + "_" + n, vtype = GRB.CONTINUOUS, lb = 0)

        for b in range(self.n_bodies):
            self.tau_a_b.append({})
            for n in self.N:
                self.tau_a_b[b][n] = self.opt_mod.addVar(name = "tau_B_a_" + str(b) + "_" + n, vtype = GRB.CONTINUOUS, lb = 0)

        for b in range(self.n_bodies):
            self.tau_d_b.append({})
            for n in self.N:
                self.tau_d_b[b][n] = self.opt_mod.addVar(name = "tau_B_d_" + str(b) + "_" + n, vtype = GRB.CONTINUOUS, lb = 0)

        for v in range(self.n_vessels):
            self.y_v.append({})
            for i in self.N:
                for j in self.N:
                    if(i == j):
                        continue
                    self.y_v[v][(i,j)] = self.opt_mod.addVar(name = "y_" + str(v) + "_(" + i + "," + j + ")", vtype = GRB.BINARY)

        for b in range(self.n_bodies):
            self.phi_b.append({})
            for i in self.N:
                for j in self.N:
                    if(i == j):
                        continue
                    self.phi_b[b][(i,j)] = self.opt_mod.addVar(name = "phi_" + str(b) + "_(" + i + "," + j + ")", vtype = GRB.BINARY)
        
    def _generate_routing_constraints(self):
        self.c_2 = []
        self.c_3 = []
        self.c_4 = {}

        for v in range(self.n_vessels):
            self.c_2.append(
                self.opt_mod.addConstr(
                    quicksum(
                        self.z_v[v][arc] for arc in self.A.select(self.J_0[v],"*")) == 1))

        for v in range(self.n_vessels):
            self.c_3.append(
                self.opt_mod.addConstr(
                    quicksum(
                        self.z_v[v][arc] for arc in self.A_sink) == 1))

        for v in range(self.n_vessels):
            for i in self.N:
                if(i == self.J_0[v] or i == "sink"):
                    continue
                self.c_4[(v,i)] = self.opt_mod.addConstr(
                    quicksum(self.z_v[v][arc] for arc in self.A.select("*",i)) == 
                    quicksum(self.z_v[v][arc] for arc in self.A.select(i,"*")))

        self.c_5 = []
        self.c_6 = []
        self.c_7 = {}

        for b in range(self.n_bodies):
            self.c_5.append(
                self.opt_mod.addConstr(
                    quicksum(
                        self.beta_b[b][arc] for arc in self.A.select(self.J_0[self.n_vessels + b],"*")) == 1))

        for b in range(self.n_bodies):
            self.c_6.append(
                self.opt_mod.addConstr(
                    quicksum(
                        self.beta_b[b][arc] for arc in self.A_sink) == 1))

        for b in range(self.n_bodies):
            for i in self.N:
                if(i == self.J_0[self.n_vessels + b] or i == "sink"):
                    continue
                self.c_7[(b,i)] = self.opt_mod.addConstr(
                    quicksum(self.beta_b[b][arc] for arc in self.A.select("*",i)) == 
                    quicksum(self.beta_b[b][arc] for arc in self.A.select(i,"*")))

        self.c_8 = {}
        self.c_9 = {}

        for b in range(self.n_bodies):
            for arc in self.A_D:
                self.c_8[(b,arc)] = self.opt_mod.addConstr(
                    self.beta_b[b][arc] == 
                    self.delta_b[b][arc] + quicksum(self.omega_bv[v][b][arc] for v in range(self.n_vessels)))

        for b in range(self.n_bodies):
            for arc in set(self.A).difference(self.A_D):
                self.c_9[(b,arc)] = self.opt_mod.addConstr(
                    self.beta_b[b][arc] == quicksum(self.omega_bv[v][b][arc] for v in range(self.n_vessels)))
            
    def _generate_sub_tour_elimination_constraints(self):
        self.c_10 = {}
        self.c_11 = {}
        self.c_12 = {}
        self.c_13 = {}

        for v in range(self.n_vessels):
            for arc in self.A:
                self.c_10[(v,arc)] = self.opt_mod.addConstr(self.z_v[v][arc] <= self.y_v[v][arc])

        for v in range(self.n_vessels):
            for i,j in [(i,j) for (i,j) in self.A if self.A.select(j,i)]:
                self.c_11[v,(i,j)] = self.opt_mod.addConstr(self.y_v[v][(i,j)] + self.y_v[v][(j,i)] == 1)

        for v in range(self.n_vessels):
            for i,j in [(i,j) for (i,j) in self.A if not self.A.select(j,i)]:
                self.c_12[v,(i,j)] = self.opt_mod.addConstr(self.y_v[v][(i,j)] == 1)

        for v in range(self.n_vessels):
            for i,j in self.A:
                for _,l in self.A.select(j,"*"):
                    if self.A.select(l,i):
                        self.c_13[v,(i,j,l)] = self.opt_mod.addConstr(self.y_v[v][(i,j)] + self.y_v[v][(j,l)] + self.y_v[v][(l,i)] <= 2)

        # Sub-tour elimination for body routes
        self.c_14 = {}
        self.c_15 = {}
        self.c_16 = {}
        self.c_17 = {}

        for b in range(self.n_bodies):
            for arc in self.A:
                self.c_14[(b,arc)] = self.opt_mod.addConstr(self.beta_b[b][arc] <= self.phi_b[b][arc])

        for b in range(self.n_bodies):
            for i,j in [(i,j) for (i,j) in self.A if self.A.select(j,i)]:
                self.c_15[b,(i,j)] = self.opt_mod.addConstr(self.phi_b[b][(i,j)] + self.phi_b[b][(j,i)] == 1)

        for b in range(self.n_bodies):
            for i,j in [(i,j) for (i,j) in self.A if not self.A.select(j,i)]:
                self.c_16[b,(i,j)] = self.opt_mod.addConstr(self.phi_b[b][(i,j)] == 1)

        for b in range(self.n_bodies):
            for i,j in self.A:
                for _,l in self.A.select(j,"*"):
                    if self.A.select(l,i):
                        self.c_17[b,(i,j,l)] = self.opt_mod.addConstr(self.phi_b[b][(i,j)] + self.phi_b[b][(j,l)] + self.phi_b[b][(l,i)] <= 2)
                        
    def _generate_capacity_constraints(self):
        self.c_18 = {}
        self.c_19 = {}

        for b in range(self.n_bodies):
            for arc in self.A:
                self.c_18[b,arc] = self.opt_mod.addConstr(
                    quicksum(
                        self.Requests["num_containers"][r]*self.x_br[r][b][arc] for r in range(self.n_requests)) <= 
                        self.Params["body_capacity"][0]*self.beta_b[b][arc])

        for v in range(self.n_vessels):
            for arc in self.A:
                self.c_19[v,arc] = self.opt_mod.addConstr(
                    quicksum(
                        self.omega_bv[v][b][arc] for b in range(self.n_bodies)) <= 
                        self.Params["max_bodies"][0]*self.z_v[v][arc])
            
    def _generate_serving_requests_constraints(self):
        self.c_20 = {}
        self.c_21 = {}
        self.c_22 = {}
        self.c_23 = {}
        self.c_24 = {}

        for i in self.Pickup_nodes + self.Delivery_nodes:
            r = self.nodes_requests[i]
            n = int(i[2:]) + self.n_requests
            copy_i = i[0] + "'_" + str(n)
            arc = self.A.select(i,copy_i)[0]
            self.c_20[i] = self.opt_mod.addConstr(
                quicksum(self.x_br[r][b][arc] for b in range(self.n_bodies)) == self.gamma_r[r])

        for b in range(self.n_bodies):
            for p in self.Pickup_nodes:
                r = self.nodes_requests[p]
                if(self.Requests["pickup_location"][r] == -1 or self.Requests["delivery_location"][r] == -1):
                    continue
                n = int(p[2:])
                copy_p   = "P'_" + str(n + self.n_requests)
                d        = "D_"  + str(n + 2*self.n_requests)
                copy_d   = "D'_" + str(n + 3*self.n_requests)
                arc_p = self.A_2.select(p,copy_p)[0]
                arc_d = self.A_4.select(d,copy_d)[0]
                self.c_21[b,i] = self.opt_mod.addConstr(self.x_br[r][b][arc_p] == self.x_br[r][b][arc_d])

        for b in range(self.n_bodies):
            for i in self.Pickup_nodes:
                r = self.nodes_requests[i]
                if(self.Requests["pickup_location"][r] == -1 or self.Requests["delivery_location"][r] == -1):
                    continue
                copy_d = "D'_" + str(int(i[2:]) + 3*self.n_requests)
                for j in self.N:
                    if(j == i or j == copy_d or j == "sink"):
                        continue
                    self.c_22[b,i,j] = self.opt_mod.addConstr(
                                                quicksum(self.x_br[r][b][self.A.select(j,k)[0]] for k in self.N if k != i and self.A.select(j,k)) ==
                                                quicksum(self.x_br[r][b][self.A.select(k,j)[0]] for k in self.N if k != copy_d and self.A.select(k,j)))

        for i in self.Delivery_nodes:
            r = self.nodes_requests[i]
            if(self.Requests["pickup_location"][r] != -1):
                continue
            b = self.Requests["at_body"][r]
            copy_i = "D'_" + str(int(i[2:]) + self.n_requests)
            arc = self.A_4.select(i,copy_i)[0]
            self.c_23[i] = self.opt_mod.addConstr(self.x_br[r][b][arc] == self.gamma_r[r])

        for i in self.N:
            for r in range(self.n_requests):
                for b1,b2 in combinations(range(self.n_bodies),r=2):
                    self.c_24[i,r,b1,b2] = self.opt_mod.addConstr(
                                                        quicksum(self.x_br[r][b1][arc] for arc in self.A.select("*",i)) +
                                                        quicksum(self.x_br[r][b2][arc] for arc in self.A.select(i,"*")) <= 1)
            
    def _generate_time_constraints(self):
        self.T_max = max(self.Requests["delivery_to"])

        # Updating arrival/departure times
        self.c_25 = {}
        self.c_26 = {}

        for v in range(self.n_vessels):
            for arc in self.A:
                self.c_25[v,arc] = self.opt_mod.addConstr(self.tau_a_v[v][arc[1]] >= 
                                                self.tau_d_v[v][arc[0]] + 
                                                self.travel_time[arc] * self.z_v[v][arc] - 
                                                self.T_max * (1 - self.z_v[v][arc]))

        for b in range(self.n_bodies):
            for arc in self.A:
                self.c_26[b,arc] = self.opt_mod.addConstr(self.tau_a_b[b][arc[1]] >= 
                                                self.tau_d_b[b][arc[0]] + 
                                                self.travel_time[arc] * self.beta_b[b][arc] - 
                                                self.T_max * (1 - self.beta_b[b][arc]))
        # Departures after arrivals
        self.c_27 = {}
        self.c_28 = {}

        for v in range(self.n_vessels):
            for i in self.N:
                self.c_27[v,i] = self.opt_mod.addConstr(self.tau_d_v[v][i] >= self.tau_a_v[v][i])

        for b in range(self.n_bodies):
            for i in self.N:
                self.c_28[b,i] = self.opt_mod.addConstr(self.tau_d_b[b][i] >= self.tau_a_b[b][i])

        # Attaching and detaching times
        self.c_29 = {}
        self.c_30 = {}

        for v in range(self.n_vessels):
            for b in range(self.n_bodies):
                for i in self.N:
                    if(i == "o_b_" + str(b) or i == "sink"):
                        continue
                    self.c_29[v,b,i] = self.opt_mod.addConstr(self.tau_d_v[v][i] >= 
                                                    self.tau_a_v[v][i] + self.Params["decoupling_time"][0] +
                                                    self.T_max * quicksum(self.omega_bv[v][b][arc] for arc in self.A.select("*",i)) -
                                                    self.T_max * quicksum(self.omega_bv[v][b][arc] for arc in self.A.select(i,"*")) -
                                                    self.T_max)

                    self.c_30[v,b,i] = self.opt_mod.addConstr(self.tau_d_v[v][i] >= 
                                                    self.tau_a_v[v][i] + self.Params["coupling_time"][0] -
                                                    self.T_max * quicksum(self.omega_bv[v][b][arc] for arc in self.A.select("*",i)) +
                                                    self.T_max * quicksum(self.omega_bv[v][b][arc] for arc in self.A.select(i,"*")) -
                                                    self.T_max)
        # Vessel arrives before body
        self.c_31 = {}
        self.c_32 = {}
        self.c_33 = {}
        self.c_34 = {}

        for v in range(self.n_vessels):
            for b in range(self.n_bodies):
                for i in self.N:
                    self.c_31[v,b,i] = self.opt_mod.addConstr(self.tau_a_b[b][i] >=
                                                    self.tau_a_v[v][i] + 
                                                    self.T_max * quicksum(self.omega_bv[v][b][arc] for arc in self.A.select("*",i)) - self.T_max)
                    self.c_32[v,b,i] = self.opt_mod.addConstr(self.tau_d_b[b][i] >=
                                                    self.tau_d_v[v][i] + 
                                                    self.T_max * quicksum(self.omega_bv[v][b][arc] for arc in self.A.select("*",i)) - self.T_max)
                    self.c_33[v,b,i] = self.opt_mod.addConstr(self.tau_a_v[v][i] >=
                                                    self.tau_a_b[b][i] + 
                                                    self.T_max * quicksum(self.omega_bv[v][b][arc] for arc in self.A.select(i,"*")) - self.T_max)
                    self.c_34[v,b,i] = self.opt_mod.addConstr(self.tau_d_v[v][i] >=
                                                    self.tau_d_b[b][i] + 
                                                    self.T_max * quicksum(self.omega_bv[v][b][arc] for arc in self.A.select(i,"*")) - self.T_max)
        # Time windows
        self.c_35 = {}
        self.c_36 = {}
        self.c_37 = {}
        self.c_38 = {}

        for b in range(self.n_bodies):
            for i in self.Pickup_nodes:
                r = self.nodes_requests[i]
                t = self.Requests["pickup_from"][r]
                self.c_35[b,i] = self.opt_mod.addConstr(self.tau_d_b[b][i] >= t * self.gamma_r[r])

            for i in self.Delivery_nodes:
                r = self.nodes_requests[i]
                t = self.Requests["delivery_from"][r]
                self.c_36[b,i] = self.opt_mod.addConstr(self.tau_d_b[b][i] >= t * self.gamma_r[r])

            for i in self.Pickup_nodes_copy:
                r = self.nodes_requests[i]
                t = self.Requests["pickup_to"][r]
                self.c_37[b,i] = self.opt_mod.addConstr(self.tau_a_b[b][i] <= t * self.gamma_r[r])

            for i in self.Delivery_nodes_copy:
                r = self.nodes_requests[i]
                t = self.Requests["delivery_to"][r]
                self.c_38[b,i] = self.opt_mod.addConstr(self.tau_a_b[b][i] <= t * self.gamma_r[r])
                
    def _generate_transfer_constraints(self):
        self.c_39 = {}
        self.c_40 = {}
        self.c_41 = {}

        for b in range(self.n_bodies):
            for v in range(self.n_vessels):
                for s in set(self.S).difference(self.S_b[b]):
                    self.c_39[b,v,s] = self.opt_mod.addConstr(
                                        quicksum(self.omega_bv[v][b][arc] for arc in self.A.select("*",s)) == 
                                        quicksum(self.omega_bv[v][b][arc] for arc in self.A.select(s,"*")))
            for s_1 in [s_b for s_b in self.S_b[b] if s_b[-1] == "1"]:
                s_2 = s_1[:-1] + "2"
                self.c_40[b,v,s_1] = self.opt_mod.addConstr(
                                        quicksum(self.beta_b[b][arc] for arc in self.A.select("*",s_1)) == self.delta_b[b][s_1,s_2])
                self.c_41[b,v,s_2] = self.opt_mod.addConstr(
                                        quicksum(self.beta_b[b][arc] for arc in self.A.select(s_2,"*")) == self.delta_b[b][s_1,s_2])
            
    def _generate_valid_inequalities(self):
        self.c_44 = {}
        self.c_45 = {}

        for i in self.Pickup_nodes:
            r = self.nodes_requests[i]
            self.c_44[i] = self.opt_mod.addConstr(quicksum(
                                            quicksum(
                                                self.x_br[r][b][arc] for arc in self.A.select("*",i))
                                            for b in range(self.n_bodies)) == 0)

        for i in self.Delivery_nodes_copy:
            r = self.nodes_requests[i]
            self.c_45[i] = self.opt_mod.addConstr(quicksum(
                                            quicksum(
                                                self.x_br[r][b][arc] for arc in self.A.select(i,"*"))
                                            for b in range(self.n_bodies)) == 0)
        
    def plot_solution(self):
        colors = [x for x in mcd.TABLEAU_COLORS.values()]
        G = nx.MultiDiGraph()
        for is_vessels in [0,1]:
            n = self.n_vessels if is_vessels else self.n_bodies
            for i in range(n):
                path = self._location_paths(i,is_vessels)
                G.add_edges_from(path)
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G,pos,node_color="#90EE90",node_size = 500)
        nx.draw_networkx_labels(G,pos)

        patches = []

        for is_vessels in [0,1]:
            n = self.n_vessels if is_vessels else self.n_bodies
            for i in range(n):
                path = self._location_paths(i,is_vessels)
                self._draw_edges(G,pos,path,i,is_vessels,colors)
                label = "Vessel " if is_vessels else "Body "
                label += str(i)
                patches.append(mpatches.Patch(color=colors[self.n_bodies * is_vessels + i], label=label))

        self._print_information()
        plt.legend(handles=patches)
        plt.show()
        
    def _get_shared_routes(self,vessel,body):
        arcs = self.omega_bv[vessel][body]
        locs = []
        for arc in arcs:
            if arcs[arc].x == 1:
                locs.append((self.nodes_locations[arc[0]],self.nodes_locations[arc[1]]))
        return locs

    def _draw_edges(self,G,pos,path,i,is_vessels,colors):
        if is_vessels:
            radius = (-1)**i * 0.5 * (i + 1)
            style  = "arc3,rad=" + str(radius)
            nx.draw_networkx_edges(G,pos,edgelist=path,width = 3,connectionstyle=style, edge_color=colors[self.n_bodies * is_vessels + i])
            return

        for arc in path:
            radius = (-1)**i * 0.1 * i
            for v in range(self.n_vessels):
                if arc in self._get_shared_routes(v,i):
                    radius = (-1)**v * 0.5 * (v + 1) - 0.05*(i + 1)
                    break
            style  = "arc3,rad=" + str(radius)
            nx.draw_networkx_edges(G,pos,edgelist=[arc],width = 3,connectionstyle=style, edge_color=colors[self.n_bodies * is_vessels + i],style="dashed")

    def _print_information(self):
        for v in range(self.n_vessels):
            print("Vessel: {:1} - start location: {}".format(v,self.Vessels["start_location"][v]))
        for b in range(self.n_bodies):
            print("Body: {:3} - start location: {}".format(b,self.Bodies["start_location"][b]))
        for r in range(self.n_requests):
            if(self.gamma_r[r].x == 0):
                print("Request",r,"outsourced by trucks!")
                continue
            r_p_l = self.Requests["pickup_location"][r]
            r_d_l = self.Requests["delivery_location"][r]
            r_p_node = "P_" + str(r)
            r_p_copy_node = "P'_" + str(r + self.n_requests)
            r_d_node = "D_" + str(r + 2 * self.n_requests)

            if r_p_l == -1:
                r_b   = self.Requests["at_body "][r]
                r_d_t = round(self.tau_a_b[r_b][r_d_node].x,2)
                print("Request",r,"handled by body",r_b,"- Delivery at location",r_d_l,"at",r_d_t)
                continue

            for b in range(self.n_bodies):
                if self.x_br[r][b][self.A.select(r_p_node,r_p_copy_node)[0]].x == 1:
                    r_p_t = round(self.tau_a_b[b][r_p_node].x,2)
                    if r_d_l == -1:            
                        print("Request",r,"handled by body",r_b,"- Pickup at location",r_p_l,"at",r_p_t)
                        continue
                    r_d_t = round(self.tau_a_b[b][r_d_node].x,2)
                    print("Request",r,"handled by body",b,"- Pickup at location",r_p_l,"at",r_p_t," - Delivery at location",r_d_l,"at",r_d_t)

    def _location_paths(self,index,is_vessels,timing=False):
        arc_list = self.z_v[index]     if is_vessels == 1 else self.beta_b[index]
        arr_time = self.tau_a_v[index] if is_vessels == 1 else self.tau_a_b[index] 
        dep_time = self.tau_d_v[index] if is_vessels == 1 else self.tau_d_b[index]
        path = [arc for arc in arc_list if arc_list[arc].x == 1]
        path_locations = []
        timings = {}
        for arc in path:
            dep = self.nodes_locations[arc[0]]
            arr = self.nodes_locations[arc[1]]
            if dep != arr and arr != -1:
                if timing:
                    t_dep = round(self.dep_time[arc[0]].x,2)
                    t_arr = round(self.arr_time[arc[1]].x,2)
                    timings[(dep,arr)] = (t_dep,t_arr)
                path_locations.append((dep,arr))
        if timing:
            return path_locations,timings
        return path_locations

