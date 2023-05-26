# Copyright © 【2023】 【Huawei Technologies Co., Ltd】. All Rights Reserved.
# An algorithm for computing conditional graph entropy

import numpy as np
import networkx as nx

def entropy(vec):
    vec_log=np.log(np.where(vec==0,1,vec))
    return -np.inner(vec,vec_log)

# axis=0: conditioned on the second variable (default)
# axis=1: conditioned on the first variable
def cond_entropy(arr,axis=0):
    return entropy(arr.flatten())-entropy(arr.sum(axis=axis))

def random_01matrix(size0,size1,prob):
    arr=np.random.rand(size0,size1)
    return np.where(arr<prob,1,0)

def random_prob_arr(shape):
    arr=np.random.rand(*shape)
    return arr/arr.sum()

def percolate(arr,prob):
    perc_arr=np.where(np.random.rand(*arr.shape)<prob,0,arr)
    return perc_arr/perc_arr.sum()

def find_ind_sets(nw,verbose_mode=False):
    if verbose_mode:
        print("running Bron–Kerbosch Algorithm to find maximal independent sets...")
    nw=nx.convert_node_labels_to_integers(nw)
    adj=nx.complement(nw).adj
    graph={v: set(adj[v]) for v in nw.nodes}
    cliques = []
    find_cliques_pivot(graph,set(),set(graph.keys()),set(),cliques)
    nr_j=len(cliques)
    nr_x=len(graph)
    sets=np.zeros((nr_j,nr_x), dtype=int)
    for j in range(nr_j):
        for x in cliques[j]:
            sets[j,x]=1
    if verbose_mode:
        print("{} maximal independent sets found".format(nr_j))
    return sets

#Bron–Kerbosch Algorithm with "pivot vertex"
def find_cliques_pivot(graph, r, p, x, cliques):
    if len(p) == 0 and len(x) == 0:
        cliques.append(r)
    else:
        deg_in_p = {u: len( p.intersection(graph[u]) ) for u in p.union(x)}
        u_max=max(deg_in_p, key=deg_in_p.get)
        for v in p.difference(graph[u_max]):
            neighs = graph[v]
            find_cliques_pivot(graph, r.union([v]), p.intersection(neighs), x.intersection(neighs), cliques)
            p.remove(v)
            x.add(v)
    
class SumProductOptimizer:
    verbose_mode=False
    #eps_prec=2.**-50
    eps_prec=0.
    eps_assert=2**-40

    def __init__(self,ga,al,be=np.empty(0)):
        self.ga=ga
        self.al=al
        self.nr_y=al.shape[0]
        self.nr_x=al.shape[1]
        assert len(ga)==self.nr_x
        assert all( abs(al.sum(axis=0)-1)<self.eps_assert )
        self.be = np.ones(self.nr_y) if len(be)==0 else be
        assert all( self.be > 0 )
        #where  al.sum(axis=0)>0 , the optimal t will be 0:
        #could set those coordinates to 0 in the first place

    def t_new(self,t):
        fx=self.ga*np.exp(np.dot(np.log(np.where(t==0,1,t)),self.al))
        t_new=(fx*self.al).sum(axis=1)
        t_new=t_new/t_new.sum()/self.be
        return (fx.sum(), t_new)
        
    def dist(self,t):
        return abs(self.t_new(t)[1]-t).sum()
            
    def find_max(self):
        t=np.ones(self.nr_y)/self.be.sum()
        steps=0
        val=-np.inf
        while True:
            steps+=1
            res=self.t_new(t)
            val_new=res[0]
            if val_new > val+self.eps_prec:
                val=val_new
                t=res[1]
            else:    
                if self.verbose_mode:
                    print("after {} steps with optimality {}".format(steps,self.dist(t)))
                    print("max value: {}   at:".format(res[0]))
                    print("at: {}".format(res[1]))
                    print()
                return res

class GraphEntropy:
    verbose_mode=False
    block=10
    steps_max=10000
    eps_prec=2.**-50
    #eps_prec=0.
    eps_active=0
    #eps_active=2.**-20
    eps_assert=2**-40

    def __init__(self,sets):
        assert all( val==0 or val==1 for val in np.nditer(sets)), "0-1 matrix is expected"
        assert all( sets.sum(axis=0)>0 ), "The sets do not cover all vertices!"
        self.nr_x=sets.shape[1]
        self.or_sets=sets
        self.sets_reset()

    def sets_reset(self):
        self.nr_j=self.or_sets.shape[0]
        self.sets=self.or_sets
        self.active_sets=np.arange(self.nr_j)
        if hasattr(self, 'p'):
            self.update_r_mask()

    def update_r_mask(self):
        self.r_mask=np.where(self.R(self.sets)>0,True,False)

    def forced_zeros(self):
        return np.count_nonzero(~self.r_mask)

    def set_p(self,p):
        assert self.nr_x==p.shape[0]
        assert abs(p.sum()-1)<self.eps_assert, "the sum of probabilities should be 1"
        if p.ndim==1:
            assert all(val>0 for val in np.nditer(p)), "probabilities should be positive"
            self.cond=False
            self.px=p
            self.py=np.ones(1)
            self.pxy=p
            self.nr_y=1
        else:
            self.cond=True
            self.nr_y=p.shape[1]
            self.p=p
            self.px=self.p.sum(axis=1)
            self.py=self.p.sum(axis=0)
            assert all(val>=0 for val in np.nditer(p)), "probabilities should be nonnegative"
            assert all(self.px>0) and all(self.py>0), "all marginals should be positive"
            self.pyx=self.p/np.reshape(self.px,(-1,1))
            self.pxy=np.transpose(self.p/self.py)
        self.update_r_mask()        

    def set_uniform_p(self):
        self.set_p( (1./self.nr_x)*np.ones(self.nr_x) )
    
    def uniform_q(self):
        arr=1.*self.sets
        return arr/arr.sum(axis=0)
        
    def uniform_r(self):
        sh=(self.nr_j,self.nr_y) if self.cond else self.nr_j
        return (1./self.nr_j)*np.ones(sh)
        
    def random_q(self):
        arr=self.sets*np.random.rand(self.nr_j,self.nr_x)
        return arr/arr.sum(axis=0)

    def random_r(self):
        arr=np.random.rand(self.nr_j,self.nr_y) if self.cond else np.random.rand(self.nr_j)
        return arr/arr.sum(axis=0)

    def phi_a(self,a):
        return -(np.log(a)*self.px).sum()

    def phi(self,q,r):
        q_log_q=q*np.log(np.where(q==0,1,q))
        val1=np.inner(q_log_q.sum(axis=0),self.px)
        r2=self.R(q)
        r2_log_r=r2*np.log(np.where(r==0,1,r)) #only works if not infty
        val2=np.sum(r2_log_r.sum(axis=0)*self.py)
        return val1-val2

    def delta(self,q1,q2):
        #only works if not infty
        #q1*log(q1/q2):
        arr=q1*(np.log(np.where(q1==0,1,q1))-np.log(np.where(q1==0,1,q2)))
        return np.inner(arr.sum(axis=0),self.px)

    #checks if q[j,x]>0 whenever x in j (i.e. sets[j,x]==1)
    def int_Kq(self,q):
        uf=np.where(self.sets==1,q>0,True)
        return all(np.nditer(uf))

    #checks if all r values are positive except forced zeros
    def int_Kr(self,r):
        uf=np.where(self.r_mask,r>0,True)
        return all(np.nditer(uf))

    #only works when r has no "unforced zeros"
    def Q(self,r):
        #may comment out this assertion (will lead to an error anyways)
        assert self.int_Kr(r), "unforced zero in r"
        gjx=np.exp( np.inner(np.log(np.where(r==0,1,r)),self.pyx) )*self.sets if self.cond else self.sets*r.reshape((-1,1))
        a=gjx.sum(axis=0)
        return gjx/a
    
    def R(self,q):
        return np.inner(q,self.pxy)
        
    def iter_step(self, st=1):
        for _ in range(st):
            gjx=np.exp( np.inner(np.log(np.where(self.r==0,1,self.r)),self.pyx) )*self.sets if self.cond else self.sets*self.r.reshape((-1,1))
            self.a=gjx.sum(axis=0)
            self.q=gjx/self.a
            #if not self.int_Kq(self.q):
            #    print("WARNING: unforced zero in q")
            self.r=np.inner(self.q,self.pxy)        

    def iter(self):
        self.steps=0
        old_val=np.inf
        new_val=np.inf
        while self.steps<self.steps_max:
            self.steps+=self.block
            self.iter_step(self.block)
            new_val=self.phi_a(self.a)
            if new_val>old_val-self.eps_prec:
                break
            old_val=new_val
            if self.eps_active>0:
                self.nullify()
        if self.verbose_mode:
            print("{} iterations made (max: {}). Current value:".format(self.steps,self.steps_max))
            print(new_val)
            
    def nullify(self):
        s=((self.r.sum(axis=1) if self.cond else self.r) > self.eps_active)
        if all(s):
            return
        deleted=self.active_sets[~s]
        self.sets=self.sets[s]
        self.nr_j=self.sets.shape[0]
        self.active_sets=self.active_sets[s]
        self.update_r_mask()        
        self.r=self.r[s]
        self.r=self.r/self.r.sum(axis=0)
        if self.verbose_mode:
            print("{} set(s) deleted after {} iterations".format(len(deleted),self.steps))
            #print(deleted)
            #print("{} forced zeros in r".format(self.forced_zeros()))
            
    def re_activate(self,re_act):
        eps=1./(1024*len(re_act)*self.nr_y)
        self.active_sets=np.concatenate((self.active_sets, re_act))
        for j in re_act:
            self.sets=np.vstack([self.sets, self.or_sets[j]])
            #one may use eps*t instead
            self.r=np.vstack([self.r,np.full((self.nr_y,),eps)])
        self.nr_j=self.sets.shape[0]
        self.update_r_mask()        
        self.r=self.r/self.r.sum(axis=0)
        if self.verbose_mode:
            print("{} set(s) reactivated: {}".format(len(re_act),re_act))    
                
    def check_set(self,s):
        mask = (s[:]==1)
        ga=self.px[mask]/self.a[mask]
        al=np.transpose(self.pyx[mask,:])
        spo=SumProductOptimizer(ga,al,self.py)
        return spo.find_max()[0]

    def opt_check(self):
        des=np.array([self.check_set(s) for s in self.or_sets]) if self.cond else np.dot(self.or_sets,self.px/self.a)
        des=des-1
        re_act=[j for j, de in enumerate(des) if de>0 and j not in self.active_sets]
        return np.amax(des),re_act
    
    def alt_opt(self,factor_active=2.**-10):
        #self.sets_reset()
        self.r=self.uniform_r()
        self.eps_active=factor_active*self.nr_y/self.nr_j
        self.iter()

        self.eps_active=0
        while True:
            de,re_act=self.opt_check()
            if len(re_act)==0:
                val=self.phi_a(self.a)
                if self.verbose_mode:
                    prefix="Conditional " if self.cond else ""
                    print(prefix+"Graph Entropy:")
                    print("{} (error bound: {})".format(val,de))
                return val,de
            #reactivating deleted sets that failed the optimality check
            self.re_activate(re_act)
            self.iter()

    def current_derivative(self):
        if self.cond:
            #only works when r has no zeros. fix:
            #return np.where(self.r==0,-self.py,-np.matmul(self.q,self.p)/self.r)
            return -np.matmul(self.q,self.p)/self.r
        else:
            return -np.dot(self.sets,self.px/self.a)
            
    def print_result(self):
        st="conditional " if self.cond else ""
        print(st+"graph entropy:")
        print(self.phi_a(self.a))
        print("versus "+st+"entropy:")
        print(cond_entropy(self.p) if self.cond else entropy(self.px))
        self.print_r()
        print("The corresponding point in K_a:")
        print(self.a)
        print()

    def print_param(self):
        print("Set system:")
        print(self.sets)
        print()
        if self.cond:
            print("Joint distribution of (X,Y):")
            print(self.p)
        else:        
            print("Distribution of X:")
            print(self.px)
        print()
        
    def print_r(self):
        print("Weights of independent sets:")
        for j in range(self.nr_j):
            print("#{}: {}".format(self.active_sets[j],self.sets[j]))
            print(self.r[j])
        print()

    #3-pt property
    def test_3pt(self):
        q=self.random_q()
        r=self.random_r()
        Qr=self.Q(r)
        aa=self.delta(q,Qr)
        bb=self.phi(Qr,r)
        cc=self.phi(q,r)
        print("3-pt property: {}+{}={} should be = {}".format(aa,bb,aa+bb,cc))
        return aa+bb-cc

    #4-pt property
    def test_4pt(self):
        q=self.random_q()
        qq=self.random_q()
        r=self.random_r()
        Rqq=self.R(qq)
        aa=self.delta(q,qq)
        bb=self.phi(q,r)
        cc=self.phi(q,Rqq)
        print("4-pt property: {}+{}={} should be >= {}".format(aa,bb,aa+bb,cc))
        return aa+bb-cc

    #5-pt property
    def test_5pt(self):
        q=self.random_q()
        r=self.random_r()
        r0=self.random_r()
        q1=self.Q(r0)
        r1=self.R(q1)
        aa=self.phi(q1,r0)
        bb=self.phi(q,r1)
        cc=self.phi(q,r)
        dd=self.phi(q,r0)
        print("strong 5-pt property: {} should be <= {}".format(aa+bb,cc+dd))
        return cc+dd-aa-bb


#USAGE: graph entropy of the dodecahedral graph and uniform distribution    

#ge=GraphEntropy(find_ind_sets(nx.dodecahedral_graph()))
#ge.set_uniform_p()
#ge.verbose_mode=True
#ge.alt_opt()


#USAGE: conditional graph entropy for [Orlitsky-Roche, Example 2]: 

#G=nx.Graph()
#G.add_nodes_from([0,1,2])
#G.add_edge(0,2)
#ge=GraphEntropy(find_ind_sets(G))
#ge.set_p( (1./6)*np.array([[0,1,1],[1,0,1],[1,1,0]]))
#ge.print_param()
#ge.alt_opt()
#ge.print_result()

