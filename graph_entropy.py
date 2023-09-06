# Copyright © 【2023】 【Huawei Technologies Co., Ltd】. All Rights Reserved.
# An algorithm for computing conditional graph entropy

import numpy as np
import networkx as nx

def entropy(vec):
    """Compute the Shannon entropy of a distribution."""
    vec_log=np.log(np.where(vec==0,1,vec))
    return -np.inner(vec,vec_log)

# axis=0: conditioned on the second variable (default)
# axis=1: conditioned on the first variable
def cond_entropy(arr,axis=0):
    """Compute the conditional entropy for a joint distribution."""
    return entropy(arr.flatten())-entropy(arr.sum(axis=axis))

def random_01matrix(size0,size1,prob):
    """Return a random 0-1 matrix of given dimensions."""
    arr=np.random.rand(size0,size1)
    return np.where(arr<prob,1,0)

def random_prob_arr(shape):
    """Return a random array with the sum of entries being 1."""
    arr=np.random.rand(*shape)
    return arr/arr.sum()

def percolate(arr,prob):
    """Zero out random entries of an array and return a normalized array (sum of entries=1)."""
    perc_arr=np.where(np.random.rand(*arr.shape)<prob,0,arr)
    return perc_arr/perc_arr.sum()

def find_ind_sets(nw,verbose_mode=False):
    """Find all maximal independent sets of a given graph and return a 0-1 array."""
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
    """
    A class for finding the maximum of a homogeneous function of degree 1.
    
    Attributes
    ----------
    nr_y : int
        number of variables
    nr_x : int
        number of monoms        
    ga : numpy array
        nonnegative coefficients of monoms; shape: (nr_x,)
    al : numpy array
        nonnegative exponents in monoms; shape: (nr_y,nr_x); the sum of each column should be 1
    be : numpy array
        maximization is performed under the condition that the inner product of the variable vector with be is 1; default: all ones
    eps_prec : float
        maximixation terminates if function value increases by at most eps_prec
    eps_assert : float
        during initialization, the sum condition is checked with error eps_assert
    verbose_mode : bool
        details of maximization are printed on screen if set to True

    Methods
    -------
    t_new(t):
        Perform one iteration of the maximization and return a new variable vector along with the corresponding (larger) function value.
    dist(t):
        Compute the L1 distance of the given vector t from the iterated vector.
    find_max:
        Repeat iterations until function value increases by at most eps_prec, then return final function value along with final variable vector.
    """
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
        """Perform one iteration of the maximization and return a new variable vector along with the corresponding (larger) function value."""
        fx=self.ga*np.exp(np.dot(np.log(np.where(t==0,1,t)),self.al))
        t_new=(fx*self.al).sum(axis=1)
        t_new=t_new/t_new.sum()/self.be
        return (fx.sum(), t_new)
        
    def dist(self,t):
        """Compute the L1 distance of the given vector t from the iterated vector."""
        return abs(self.t_new(t)[1]-t).sum()
            
    def find_max(self):
        """Repeat iterations until function value increases by at most eps_prec, then return final function value along with final variable vector."""
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
    """
    A class for computing (conditional) graph entropy via alternating optimization.
    
    Based on the algorithm outlined in the following paper:
    
    Viktor Harangi, Xueyan Niu, Bo Bai
    Conditional graph entropy as an alternating minimization problem
    https://arxiv.org/pdf/2209.00283.pdf
    
    Attributes
    ----------
    nr_x : int
        number of X values        
    nr_y : int
        number of Y values; nr_y=1 whenever cond==False
    nr_j : int
        number of active sets
    or_sets : numpy array
        0-1 array describing the sets; shape: (? ,nr_x)
    sets : numpy array
        0-1 array corresponding to active sets (rows of redundant sets deleted from or_sets); shape: (nr_j,nr_x)
    active_sets : numpy array
        array consisting of indices of active sets
    cond : bool
        True in conditional setting; False in the unconditioned setting of graph entropy
    p : numpy array
        probabilities in the joint distribution of (X,Y); only if cond==True; shape: (nr_x,nr_y)
    px : numpy array
        probabilities in the distribution of X; shape: (nr_x,)
    py : numpy array
        probabilities in the distribution of Y; shape: (nr_y,)        
    pxy : numpy array
        conditional probabilities of X|Y; shape: (nr_y,nr_x)
    pyx : numpy array
        conditional probabilities of Y|X; shape: (nr_x,nr_y)        
    q : numpy array
        current 'q' vector during iterations; shape: (nr_j,nr_x) 
    r : numpy array
        current 'r' vector during iterations; shape: (nr_j,nr_y) or (nr_j,)
    a : numpy array
        current 'a' vector during iterations; shape: (nr_x,)
    r_mask : numpy array
        it shows the places where 'r' may have nonzero entries; shape: (nr_j,nr_y)    
    block : int
        number of iterations performed in one block
    steps_max : int
        maximum number of iteration steps performed
    eps_prec : float
        minimization terminates if function value decreases by at most eps_prec
    eps_assert : float
        when setting the distribution, the sum condition is checked with error eps_assert
    eps_active : float
        threshold for deleting a set from active sets
    re_act_factor : float
        a reactivation parameter
    verbose_mode : bool
        various details of the optimization are printed on screen if set to True
        
    USAGE
    -----
    Initialize by providing the set of independent sets, as a 0-1 array of shape (nr_j,nr_x).
    Then set the (joint) distribution by using 'set_p(p)' or 'set_uniform_p()'.
    Call 'alt_opt()' to run alternating optimization.
    
    Example 1
    ---------
    Graph entropy for the dodecahedral graph and uniform distribution:    

    ge=GraphEntropy(find_ind_sets(nx.dodecahedral_graph()))
    ge.set_uniform_p()
    ge.verbose_mode=True
    ge.alt_opt()

    Example 2
    ---------
    Conditional graph entropy for [Orlitsky-Roche, Example 2]: 
    
    G=nx.Graph()
    G.add_nodes_from([0,1,2])
    G.add_edge(0,2)
    ge=GraphEntropy(find_ind_sets(G))
    ge.set_p( (1./6)*np.array([[0,1,1],[1,0,1],[1,1,0]]))
    ge.print_param()
    ge.alt_opt()
    ge.print_result()

    """
    verbose_mode=False
    block=10
    steps_max=10000
    eps_prec=2.**-50
    #eps_prec=0.
    eps_active=0
    #eps_active=2.**-20
    eps_assert=2**-40
    re_act_factor=1.

    def __init__(self,sets):
        assert all( val==0 or val==1 for val in np.nditer(sets)), "0-1 matrix is expected"
        assert all( sets.sum(axis=0)>0 ), "The sets do not cover all vertices!"
        self.nr_x=sets.shape[1]
        self.or_sets=sets
        self.sets_reset()

    def sets_reset(self):
        """Revert to the original list of sets (i.e., every set is active)."""
        self.nr_j=self.or_sets.shape[0]
        self.sets=self.or_sets
        self.active_sets=np.arange(self.nr_j)
        if hasattr(self, 'px'):
            self.update_r_mask()

    def update_r_mask(self):
        """Update r_mask."""
        self.r_mask=np.where(self.R(self.sets)>0,True,False)

    def forced_zeros(self):
        """Return the number of forced zeros in an 'r' vector."""
        return np.count_nonzero(~self.r_mask)

    def set_p(self,p):
        """Set the distribution of X or the joint distribution of (X,Y) as given."""
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
        """Set the distribution of X to be uniform (unconditioned setting)."""
        self.set_p( (1./self.nr_x)*np.ones(self.nr_x) )
    
    def uniform_q(self):
        """Return a uniform 'q' vector."""
        arr=1.*self.sets
        return arr/arr.sum(axis=0)
        
    def uniform_r(self):
        """Return a uniform 'r' vector."""
        sh=(self.nr_j,self.nr_y) if self.cond else self.nr_j
        return (1./self.nr_j)*np.ones(sh)
        
    def random_q(self):
        """Return a random 'q' vector."""
        arr=self.sets*np.random.rand(self.nr_j,self.nr_x)
        return arr/arr.sum(axis=0)

    def random_r(self):
        """Return a random 'r' vector."""
        arr=np.random.rand(self.nr_j,self.nr_y) if self.cond else np.random.rand(self.nr_j)
        return arr/arr.sum(axis=0)

    def phi_a(self,a):
        """Compute the function value at the given 'a' vector."""
        return -(np.log(a)*self.px).sum()

    def phi(self,q,r):
        """Compute the function value at the given pair of 'q' vector."""    
        q_log_q=q*np.log(np.where(q==0,1,q))
        val1=np.inner(q_log_q.sum(axis=0),self.px)
        r2=self.R(q)
        r2_log_r=r2*np.log(np.where(r==0,1,r)) #only works if not infty
        val2=np.sum(r2_log_r.sum(axis=0)*self.py)
        return val1-val2

    def delta(self,q1,q2):
        """Compute the 'squared distance' delta function of 'q' vectors.
        
        It only works if result is not infty."""    
        #q1*log(q1/q2):
        arr=q1*(np.log(np.where(q1==0,1,q1))-np.log(np.where(q1==0,1,q2)))
        return np.inner(arr.sum(axis=0),self.px)

    def int_Kq(self,q):
        """Check if q[j,x]>0 whenever x in j (i.e., whenever sets[j,x]==1)."""
        uf=np.where(self.sets==1,q>0,True)
        return all(np.nditer(uf))

    def int_Kr(self,r):
        """Check if all 'r' values except forced zeros are positive ."""
        uf=np.where(self.r_mask,r>0,True)
        return all(np.nditer(uf))

    def Q(self,r):
        """Return Q(r), see the article for the formula.
        
        It only works when r has no unforced zeros."""
        assert self.int_Kr(r), "unforced zero in r"  #may comment out this assertion (will lead to an error anyways)
        gjx=np.exp( np.inner(np.log(np.where(r==0,1,r)),self.pyx) )*self.sets if self.cond else self.sets*r.reshape((-1,1))
        a=gjx.sum(axis=0)
        return gjx/a
    
    def R(self,q):
        """Return R(q), see the article for the formula."""
        return np.inner(q,self.pxy)
        
    def iter_step(self, st=1):
        """Perform given number of iterations of alternating optimization.""" 
        for _ in range(st):
            gjx=np.exp( np.inner(np.log(np.where(self.r==0,1,self.r)),self.pyx) )*self.sets if self.cond else self.sets*self.r.reshape((-1,1))
            self.a=gjx.sum(axis=0)
            self.q=gjx/self.a
            #if not self.int_Kq(self.q):
            #    print("WARNING: unforced zero in q")
            self.r=np.inner(self.q,self.pxy)        

    def iter(self):
        """Perform iterations in blocks until function value decreases by at most eps_prec or steps_max has been reached.
        
        Sets with 'r' values under threshold are deleted from active sets along the way."""
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
            print("{} iterations made (max: {})".format(self.steps,self.steps_max))
            #print("Current value: {}".format(new_val))
            
    def nullify(self):
        """Delete sets with current 'r' values under threshold from active sets."""
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
            nr_d=len(deleted)
            list_d=': '+' '.join([self.set2str(self.or_sets[j]) for j in deleted]) if nr_d<11 else ''
            print("{} set{} deleted after {} iterations{}".format(nr_d,"s" if nr_d>1 else "",self.steps,list_d))
            #print("{} forced zeros in r".format(self.forced_zeros()))
            
    def re_activate(self,re_act):
        """Re-activate sets of the given indices.
        
        Put the corresponding rows of or_sets back to sets, using small 'r' values."""
        eps=1./(1024*len(re_act))
        self.active_sets=np.concatenate((self.active_sets, re_act))
        for j in re_act:
            self.sets=np.vstack([self.sets, self.or_sets[j]])
            self.r=np.append(self.r,eps*self.check_set(self.or_sets[j])[1],axis=0) if self.cond else np.append(self.r, np.array([eps]))
        self.nr_j=self.sets.shape[0]
        self.update_r_mask()        
        self.r=self.r/self.r.sum(axis=0)
        if self.verbose_mode:
            print("{} set(s) reactivated: {}".format(len(re_act),re_act))    
                
    def check_set(self,s):
        """Return the maximum in the dual problem (see the article) corresponding to the given set."""
        mask = (s[:]==1)
        ga=self.px[mask]/self.a[mask]
        al=np.transpose(self.pyx[mask,:])
        spo=SumProductOptimizer(ga,al,self.py)
        return spo.find_max()

    def opt_check(self):
        """Check which sets violate (and by how much) the optimality check (i.e., dual problem).
        
        Return error bound, error bound for the active part, and the indices of sets to be re-activated."""
        des=np.array([self.check_set(s)[0] for s in self.or_sets]) if self.cond else np.dot(self.or_sets,self.px/self.a)
        des=des-1
        de_act=np.amax(des[self.active_sets])
        re_act=[j for j, de in enumerate(des) if de>de_act*self.re_act_factor and j not in self.active_sets]
        return np.amax(des),de_act,re_act
    
    def alt_opt(self,factor_active=2.**-10):
        """Perform alternating optimization and compute error bound.
        
        Set deletions are performed along the way if factor_active is nonzero.
        In the end: optimality check and potential re-activation of sets."""
        #self.sets_reset()
        self.r=self.uniform_r()
        self.eps_active=factor_active*self.nr_y/self.nr_j
        self.iter()

        self.eps_active=0
        while True:
            de,de_act,re_act=self.opt_check()
            if len(re_act)==0:
                val=self.phi_a(self.a)
                if self.verbose_mode:
                    prefix="Conditional " if self.cond else ""
                    print(prefix+"Graph Entropy:")
                    print("{} (error bound: {:.1e})".format(val,de))
                return val,de
            #reactivating deleted sets that failed the optimality check
            self.re_activate(re_act)
            self.iter()

    def current_derivative(self):
        """Compute the gradient at the current point during iterations."""
        if self.cond:
            #only works when r has no zeros. fix:
            #return np.where(self.r==0,-self.py,-np.matmul(self.q,self.p)/self.r)
            return -np.matmul(self.q,self.p)/self.r
        else:
            return -np.dot(self.sets,self.px/self.a)
            
    def set2str(self,s):
        """Return the word of labels or the sequence of indices corresponding to a given set s."""
        if hasattr(self, 'lbl'):
            return ''.join([self.lbl[i] for i in range(len(s)) if s[i]==1])
        else:
            return '{'+','.join([str(i) for i in range(len(s)) if s[i]==1])+'}'

    def print_result(self):
        """Print the (cond) graph entropy compared to (cond) entropy, and the optimal 'a' vector on screen."""
        st="conditional " if self.cond else ""
        print(st+"graph entropy:")
        print(self.phi_a(self.a))
        print("versus "+st+"entropy:")
        print(cond_entropy(self.p) if self.cond else entropy(self.px))
        self.print_r()
        print("The corresponding point in K_a:")
        print(self.a)
        print()

    def print_sets(self, only_active=False):
        """Print the (active) sets on screen."""
        ss=self.sets if only_active else self.or_sets
        print("{} {}sets:".format(len(ss),"active " if only_active else ""))
        for s in ss:
            print(self.set2str(s))
        print()
        
    def print_distr(self):
        """Print the distribuiion of X or the joint distribution of (X,Y) on screen.""" 
        if self.cond:
            print("Joint distribution of (X,Y):")
            print(self.p)
        else:        
            print("Distribution of X:")
            print(self.px)
        print()
        
    def print_r(self):
        """Print the current r values on screen.""" 
        print("{} active sets (and their r values):".format(len(self.sets)))
        for j in range(self.nr_j):
            print(self.set2str(self.sets[j]))
            print(self.r[j])
        print()

    def test_3pt(self):
        """Check if 5-pt property holds for randomly chosen points."""
        q=self.random_q()
        r=self.random_r()
        Qr=self.Q(r)
        aa=self.delta(q,Qr)
        bb=self.phi(Qr,r)
        cc=self.phi(q,r)
        print("3-pt property: {}+{}={} should be = {}".format(aa,bb,aa+bb,cc))
        return aa+bb-cc

    def test_4pt(self):
        """Check if 4-pt property holds for randomly chosen points."""
        q=self.random_q()
        qq=self.random_q()
        r=self.random_r()
        Rqq=self.R(qq)
        aa=self.delta(q,qq)
        bb=self.phi(q,r)
        cc=self.phi(q,Rqq)
        print("4-pt property: {}+{}={} should be >= {}".format(aa,bb,aa+bb,cc))
        return aa+bb-cc

    def test_5pt(self):
        """Check if 5-pt property holds for randomly chosen points."""
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


