#!/usr/bin/env python
# encoding: utf-8
import abc
import math
import random
import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import erfc

__all__ = ["ModelMeta", "MoralAgentModel", "MFPModel"]

class ModelMeta(object):
    __metaclall__ = abc.ABCMeta
    
    @abc.abstractproperty
    def size(self):
        pass
    
    @abc.abstractproperty
    def state(self):
        pass
    
    @state.setter
    def state(self, proposition):
        pass
    
    @abc.abstractmethod
    def logp(self, proposition=None):
        pass
    
    @abc.abstractmethod
    def measure(self):
        pass
    
    @abc.abstractmethod
    def propose(self):
        pass
    
    @abc.abstractproperty
    def parameters(self):
        pass

class MoralAgentModel(ModelMeta):
    sqrt2 = math.sqrt(2)
    sqrt2pi = math.sqrt(2*math.pi)    
        
    def __init__(self, beta, rho, epsilon, graph,
                 agent_complexity, symmetry_breaking_direction=None,
                 delta_state=0.1, delta_weight=0.5):
        
        self._sqrtD = math.sqrt(agent_complexity)
        self._norm = self._sqrtD
        self._gamma = math.sqrt(1-rho*rho)/rho
        
        self.beta = beta
        self.rho = rho
        self.epsilon = epsilon
        self.parameter_names = ("beta","rho","epsilon")
        self.network = graph.to_directed()
        self.agent_complexity = agent_complexity
        
        self.delta_state = delta_state
        self.delta_weight = delta_weight
        
        if symmetry_breaking_direction is None:
            symmetry_breaking_direction = [1.0]*agent_complexity
            
        self.sbd = symmetry_breaking_direction
        self.zeitgeist = np.asarray(self.sbd) * self._norm / np.linalg.norm(self.sbd)
        self.D = agent_complexity
        self.N = graph.order()
        
        for i in self.network.nodes_iter():
            w = MoralAgentModel.sphere_rand(agent_complexity, self._norm)
            self.network.node[i]["vector"] = w.copy()
            self.network.node[i]["opinion"] = np.dot(w,self.zeitgeist) / self._sqrtD
            
        for i,j in self.network.edges_iter():
            self.network[i][j]["weight"] = np.random.normal(loc=0.5, scale=0.05)    
    @property
    def size(self):
        return self.N
     
    @property
    def parameters(self):
        b,r,e = self.beta, self.rho, self.epsilon
        return dict(zip(self.parameter_names,(b,r,e)))
        
    @staticmethod
    def sphere_rand(dim, norm, scale=1.0):
        rv = np.random.multivariate_normal(np.zeros(dim), scale*np.eye(dim))
        return rv * norm / np.linalg.norm(rv)

        
    @property
    def state(self):
        w = np.vstack(nx.get_node_attributes(self.network, "vector").values())
        A = np.asarray(nx.attr_matrix(self.network, edge_attr="weight")[0])
        return {"vectors":w, "adjacency":A}
    
    @state.setter
    def state(self, proposition):
        agent = proposition["agent"]
        neig = proposition["neighbor"]
        self.network.node[agent["label"]].update(agent["new_state"])
        self.network[agent["label"]][neig["label"]]["weight"] = proposition["link"]["new"]
        
    def propose(self):
        i = np.random.choice(self.N)
        ei = self.network.edges(i, data=True)
        ni,Ai = zip(*[(j,Aij["weight"]) for _,j,Aij in ei])
        pij = np.array(Ai)
        pij /= pij.sum()
        j = np.random.choice(ni, p=pij)
        state_i = self.network.node[i]
        opinion_i = state_i["opinion"]
        opinion_j = self.network.node[j]["opinion"]
        x = np.sign(opinion_j * opinion_i)
        Aij = Aij0 = self.network[i][j]["weight"]
        Aij = Aij + Aij*(1-Aij)*x*self.delta_weight
        w = state_i["vector"] + MoralAgentModel.sphere_rand(self.D, self._norm,
                                                            self.delta_state)
        w *= self._norm / np.linalg.norm(w)
        h = np.dot(w, self.zeitgeist) / self._sqrtD
        new_state = {"vector":w, "opinion":h}
        
        proposition = {
            "agent":{"label":i,
                     "state":state_i,
                     "new_state": new_state},
            "neighbor":{"label":j,
                        "opinion":opinion_j},
            "link":{"old":Aij0, "new":Aij}
        }
        return proposition
    
    @staticmethod
    def hamiltonian(hi, hj, Aij, Gamma, epsilon):
        x = hi*np.sign(hj)/Gamma/MoralAgentModel.sqrt2
        Jij = np.sign(Aij - 1/2)
        A = Jij*MoralAgentModel.sqrt2pi
        return -A*Gamma*np.log(epsilon + (1-2*epsilon)*erfc(-x)/2)
    
    def logp(self, proposition):
        hi0 = proposition["agent"]["state"]["opinion"]
        hi = proposition["agent"]["new_state"]["opinion"]
        hj = proposition["neighbor"]["opinion"]
        Aij = proposition["link"]["new"]
        Aij0 = proposition["link"]["old"]
        E0 = MoralAgentModel.hamiltonian(hi0, hj, Aij0, self._gamma,
                                         self.epsilon)
        E = MoralAgentModel.hamiltonian(hi, hj, Aij, self._gamma,
                                         self.epsilon)
        BdE = self.beta * (E - E0)
        return BdE
        
    def measure(self):
        h = np.array(nx.get_node_attributes(self.network, "opinion").values())
        m = h.mean()
        v = h.var()
        A = np.asarray(nx.attr_matrix(self.network, edge_attr="weight")[0])
        q = (A*h*h[:,None]).mean()
        Enemies = np.where(A > 1/2, 0, 1)
        n = Enemies.mean()
        data = pd.Series([m,v,q,n],index=["m","v","q","n"])
        return data
    
class MFModel(MoralAgentModel):
    @staticmethod
    def hamiltonian(hi, hj, Aij, rho, epsilon):
        Jij = np.sign(Aij - 1/2)
        v = -hi*hj + rho*abs(hi*hj)/2 - rho*abs(hi*hj+(1-2*epsilon))/2 
        return Jij*v