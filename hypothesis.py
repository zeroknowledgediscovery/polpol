import pygraphviz as pgv
import networkx as nx
import numpy as np
import re
import sys
import os
from scipy import stats
import warnings
import pandas as pd
import glob
from tqdm import tqdm
warnings.filterwarnings('ignore')

class Hypothesis(object):
    """Generate and analyze hypotheses from models inferred.  Assume gsss_timestamp format is gss_timestamp.
       Also assume that dotname or decision tree name format is gss_timestamp.dot

       Mathematical model of causal hypothesis:

       ---

       ```
       Local Marginal Regulation Coefficient: Aiming to estimate the up-regulatory/down-regulatory influence of a source organism/entity on a target organism/entity, where regulation effects are causally localized in time (future cannot affect the past) with limited memory, and potential confounding effects from other entities/organisms are marginalized out.
       ```

       Let us assume a general dependency between stochastic processes \(\\nu,u, \omega \) :

       $$ \\nu_t = \\phi(u_{\leftarrow t},\omega_{\leftarrow t}) $$

      We estimate the sign of \( \\alpha_t\) in a locally linear marginalized relationship \( \\nu_t = \\alpha_t u_{t'} + c \) with \(t' \in [ t-\delta, t] \) as follows:

    Attributes:
       qnet_orchestrator (qgss.QnetOrchestrator): instance of qgss.QnetOrchestrator with trained qnet model
       model_path (str, optional): path to directory containing generated decision trees in dot format (Default value = None)
       no_self_loops (bool, optional): If True do not report self-loops in hypotheses  (Default value = True)
       causal_constraint (float, optional): lag of source inputs from target effects. >= 0 is causal  (Default value = 0)
       total_samples (int, optional): total number of samples used to construct decision model  (Default value = 100)
       detailed_labels (bool, optional): if True, decision tree models have detailed output  (Default value = False)
       MAPNAME (str): path to dequantization map

    """

####    def __init__(self,qnet_orchestrator,
#    model_path=None,
#    no_self_loops=True,
#    causal_constraint=0,
#    total_samples=100,
#    detailed_labels=False):

    def __init__(self,
                 model_path=None,
                 no_self_loops=True,
                 total_samples=100,
                 detailed_labels=False):

        self.model_path = model_path

        variable_bin_map = dict()

        tag_list = ['abany','abdefctw', 'abdefect', 'abhlth', 'abnomore', 'abpoor', 'abpoorw',
            'abrape', 'absingle','bible','colcom','colmil','comfort','conlabor','godchnge','grass','gunlaw','intmil',
            'libcom','libmil','libhomo','libmslm','maboygrl','owngun','pillok','pilloky','polabuse','pray','prayer',
            'prayfreq', 'religcon','religint','reliten','rowngun','shotgun','spkcom','spkmil','taxrich','viruses'
        ]

        ## All answers are either left-leaning (-1) or right-leaning (+1).
        # Thus, similarly to the variable_bin_map for the quantizer, I can create
        # arrays with two numbers for each parameter analyzed in the GSS survey
        # and included in the list above.

        for param in tag_list:
            # Prameters with clear agree / disagree opinions
            if param in ["comfort", 'pillok','pilloky', 'religcon','religint','religint']:
                variable_bin_map[param] = np.array([0, 1])
            # Parameters with strong indications of frequency or opinion, but no explicit
            # 'strongly agree' / 'strongly disagree'. Examples responses
            #  - 'never'/'always', 'very good', 'very bad' 
            if param in ['abdefctw',"abpoor","bible", "godchnge", "intmil", "pray", "prayfreq", "viruses"]:
                variable_bin_map[param] = np.array([0.2, 0.8])
            # Parameters with clear positioning, but no indication of intensity. Ex: "not fired", "fired" 
            elif param in ['colcom',"colmil", "conlabor", "grass", 'libcom','libmil','libhomo',
            'libmslm', 'spkcom','spkmil','taxrich']:
                variable_bin_map[param] = np.array([0.3, 0.7])
            # Yes / No, Support / Don't support options
            else:
                variable_bin_map[param] = np.array([0.4, 0.6])
        
        self.NMAP = variable_bin_map

        self.gsss = list(set([x for x in self.NMAP]))

        self.total_samples = total_samples
        self.detailed_labels = detailed_labels

        self.decision_tree = None
        self.tree_labels = None
        self.tree_edgelabels = None
        self.TGT = None
        self.SRC = None
        self.no_self_loops = no_self_loops
        self.hypotheses=pd.DataFrame(columns=['src','tgt','lomar','pvalue'])
    
    def deQuantizer(self, name):
        """Dequantizer function

        Args:
            name: name to be verified on self.NMAP keys. If corresponding to
            no keys, then quantize based on response.

        Returns:
          float: median of dequantized value

        """
        vals = []
        splitted = name.split()

        if 'strong' in name or 'always' in name or 'never' in name:
            intensity = 1.5
        else:
            intensity = 1

        if name in self.gsss:
            vals = [r for r in self.NMAP[name]]
        elif 'no' in splitted and 'yes' not in splitted:
            vals.append(0.6*intensity)
        elif 'yes' in splitted and 'no' not in splitted:
            vals.append(0.4*(intensity - 1))
        elif 'agree' in splitted and 'disagree' not in splitted:
            vals.append(0.4*(intensity - 1))
        elif 'disagree' in splitted and 'agree' not in splitted:
            vals.append(0.6*intensity)
        elif 'should' in splitted and 'not' not in splitted:
            vals.append(0.8*intensity)
        elif 'should' in splitted and 'not' in splitted:
            vals.append(0.2*(intensity - 1))
        else:
            vals.append(0.5)

        return np.median(vals)


    def getNumeric_internal(self,
               dict_id_reached_by_edgelabel):
        """Dequantize labels on graph non-leaf nodes

        Args:
          dict_id_reached_by_edgelabel (dict[int,list[str]]): dict mapping nodeid to array of names with str type

        Returns:
          dict[int,float]: dict mapping nodeid to  dequantized values of float type

        """
        R={}
        for (k,v) in dict_id_reached_by_edgelabel:
            R[k]=np.median(
                np.array([self.deQuantizer(str(x).strip()) for x in v]))
        return R


    def getNumeric_at_leaf(self,
                    Probability_distribution_dict,
                    Sample_fraction):
        """Dequantize labels on graph leaf nodes to return mean and sample standard deviation of outputs

        Args:
          Probability_distribution_dict (dict[int, numpy.ndarray[float]]): dict mapping nodeid to probability distribution over output labels at that leaf node
          Sample_fraction (dict[int,float]): dict mapping nodeids to sample fraction captured by that leaf node

        Returns:
          float,float: mean and sample standard deviation

        """

        # ----------------------------------------
        # Q is 1D array of dequantized values
        # corresponding to levels for TGT
        # ----------------------------------------

        mux=0
        varx=0
        for (k, p) in Probability_distribution_dict:

            Q=np.array([self.deQuantizer(
            str(x).strip())
                    for x in p]).reshape(
                            len(p),1)

            mu_k=np.dot(p.transpose(),Q)
            var_k=np.dot(p.transpose(),(Q*Q))-mu_k*mu_k

            mux += Sample_fraction[k]*mu_k
            varx += Sample_fraction[k]*var_k

        return mux,np.sqrt(varx/self.total_samples)


    def regularize_distribution(self,prob,l,LABELS,e=0.005):
        """Regularize probability distribution
           using exponential decay to map non-detailed output of a
           single maximum likelihood label to a probability distribution.
           Used when detailed output is not available.

        Args:
          prob (float): probability of single output label of type str
          e (float, optional): small value to regularize return probability of 1.0 (Default value = 0.005)
          l (str): output label
          LABELS: list of possible answers to GSS parameter analyzed  

        Returns:
          numpy.ndarray: probability distribution

        """
        labels=np.array(list(LABELS))
        yy=np.ones(len(labels))*((1-prob-e)/(len(labels)-1))
        yy[np.where(labels==l)[0][0]]=prob-e
        dy=pd.DataFrame(yy).ewm(alpha=.8).mean()
        dy=dy/dy.sum()
        return dy.values


    def leaf_output_on_subgraph(self,nodeset):
        """Find the mean and sample standard deviation of output
           in leafnodes reachable from nodeset, along with fraction of samples
           captures by this subgraph

        Args:
          nodeset (numpy.ndarray): 1D array of nodeids

        Returns:
          tuple(float,float), float: mean, sample standard deviation and sample fraction

        """

        ## cLeaf is the set of leaf nodes reachable from nodeset
        # oLabels is the output labels for target and
        # is a dict mapping leafnode id to output label letter
        #
        # frac and prob are sample fraction and probability of
        # output label in leaf node, parsed from dotfile
        #
        # SUM is the total sample fraction captured by nodeset
        cLeaf=[x for x in nodeset
               if self.decision_tree.out_degree(x)==0
               and self.decision_tree.in_degree(x)==1]
        oLabels={k:str(v.split('\n')[0])
                 for (k,v) in self.tree_labels.items()
                 if k in cLeaf}
        
        total_labels = dict()
        for (k,v) in self.tree_labels.items():
            if k in cLeaf:
                initial = v.split('\n')[1].split(' ')
                total_labels[k] = [a.split(':')[0] for a in initial
                if initial.index(a) != 0]

        frac={k:float(v.split('\n')[2].replace('Frac:',''))
              for (k,v) in self.tree_labels.items()
              if k in cLeaf}
        if not self.detailed_labels:
            prob={k:float(v.split('\n')[1].split(oLabels[k]+':')[1].split(' ')[0])
                  for (k,v) in self.tree_labels.items()
                  if k in cLeaf}

            ## Get a kernel based distribution here.
            # self.alphabet=['A',...,'E']
            # prob is regularize_distributioned to get a dict {nodeid: [p1,..,pm]}
            prob__={k:self.regularize_distribution(prob[k],oLabels[k], total_labels[k])
                    for k in prob}
            prob=prob__
        else:
            prob={k:self.get_vector_from_dict(v.split('\n')[1].replace('Prob:',''))
                  for (k,v) in self.tree_labels.items()
                  if k in cLeaf}

        SUM=np.array(frac.values()).sum()

        ## mean and sample estimate of standard deviation
        mu_X,sigma_X=self.getNumeric_at_leaf(prob,frac)
        return (mu_X,sigma_X),SUM


    def getHypothesisSlice(self,nid):
        """Generating impact of node nid with source label.

        Args:
          nid (int): nodeid

        Returns:
          [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html): dataframe of hypotheses fragment with xvalue, ymean and y std dev

        """

        cNodes=list(nx.descendants(
            self.decision_tree,nid))
        nextNodes=nx.neighbors(
            self.decision_tree,nid)

        nextedge={}
        edgeProp={}
        SUM=0.
        for nn in list(nextNodes):
            nextedge[nn]=[str(x) for x in
                          self.tree_edgelabels[(
                              nid,nn)].split('\\n')]
            if len(list(nx.descendants(
                    self.decision_tree,nn))) == 0:
                res,s=self.leaf_output_on_subgraph([nn])
            else:
                res,s=self.leaf_output_on_subgraph(list(
                    nx.descendants(self.decision_tree,nn)))
            edgeProp[nn]=res
            SUM=SUM+list(s)[0]

        # nextedge is dict: {nodeid nn: letter array by which child nn is reached}
        num_nextedge=self.getNumeric_internal(
            nextedge)
        for (k,v) in edgeProp.items():
            num_nextedge[k]=np.append(
                num_nextedge[k],[v[0],v[1]])

        RF=pd.DataFrame(num_nextedge)
        RF.index=['x', 'y','sigmay']
        return RF


    def createTargetList(self,
                      source,
                      target):
        """Create list of decision trees available within time points in model_path

        Args:
          source (str): source name in gss_timestamp format
          target (str): target name in gss_timestamp format

        Returns:
          list[str]: list of paths to decision tree models in qnet

        """
        self.TGT = target
        self.SRC = source

        if self.model_path is not None:
            decision_trees = glob.glob(
                os.path.join(self.model_path,
                             target)+'*.dot')
            decision_trees_ = [x for x in decision_trees]
        else:
            raise Exception('self.model_path is not set')
        return decision_trees_


    def get_lowlevel(self,
            source,
            target):
        """Low level evaluation call to estimate local marginal regulation  \( \\alpha \)

        Args:
          source (str): source
          target (str): target

        Returns:

        """
        self.TGT = target
        self.SRC = source

        decision_trees = self.createTargetList(
            source,
            target)

        grad_=[]
        # can we do this in parallel
        for tree in decision_trees:
            self.TGT = os.path.basename(tree).replace('.dot','')
            gv = pgv.AGraph(tree,
                            strict=False,
                            directed=True)

            self.decision_tree = nx.DiGraph(gv)

            self.tree_labels = nx.get_node_attributes(
                self.decision_tree,'label')
            self.tree_edgelabels = nx.get_edge_attributes(
                self.decision_tree,"label")

            nodes_with_src=[]
            for (k,v) in self.tree_labels.items():
                if self.SRC in v:
                    nodes_with_src=nodes_with_src+[k]

            if len(nodes_with_src)==0:
                continue

            RES=pd.concat([self.getHypothesisSlice(i).transpose()
                           for i in nodes_with_src])

            grad,pvalue=self.getAlpha(RES)
            #RES.to_csv('tmp.csv')
            #if RES.index.size > 2:
            #    quit()

            #grad=stats.linregress(
            #    RES.x_.values,
            #    RES.muy.values).slope

            if np.isnan(grad):
                warnings.warn(
                    "Nan encountered in causal inferrence")
                grad=np.median(
                    RES.y.values)/np.median(
                        RES.x.values)

            self.hypotheses = self.hypotheses.append(
                {'src':self.SRC,
                 'tgt': self.TGT,
                 'lomar':float(grad),
                 'pvalue':pvalue},
                ignore_index = True)
        return


    def get(self,
            source=None,
            target=None):
        """Calculate local marginal regulation  \( \\alpha \). When source or target is not specified, we calculate for all entities available on model path. Populates self.hypotheses.

        Args:
          source (str, optional): source (Default value = None)
          target (str, optional): target (Default value = None)

        Returns:

        """

        if source is None:
            source = self.gsss
        else:
            if isinstance(source,str):
                source=[source]
        if target is None:
            target = self.gsss
        else:
            if isinstance(target,str):
                target=[target]

        for tgt_gss_ in tqdm(target):
            for src_gss_ in source:
                if (src_gss_ == tgt_gss_) and self.no_self_loops:
                    continue
                self.get_lowlevel(src_gss_,
                          tgt_gss_)

        if self.no_self_loops:
            self.hypotheses=self.hypotheses[~(self.hypotheses.src==self.hypotheses.tgt)]

        return


    def to_csv(self, *args, **kwargs):
        """Output csv of hypotheses inferred. Arguments are passed to pandas.DataFrame.to_csv()

        Args:
          *args: optional arguments to [pandas.to_csv()]( https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html)
          **kwargs: optional keywords to [pandas.to_csv()]( https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html)

        Returns:

        """
        self.hypotheses.to_csv(*args, **kwargs)


    def to_dot(self,filename='tmp.dot',
               hypotheses=None,
               square_mat=False):
        """Output dot file of hypotheses inferred.

        Args:
          filename (str, optional): filename of dot outpt (Default value = 'tmp.dot')
          hypotheses ([pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), optional): If provided use this instead of self.hypotheses (Default value = None)
          square_mat (bool, optional): If True resturn a heatmap matrix as filename+'sq.csv' (Default value = False)

        Returns:

        """

        if hypotheses is None:
            df=self.hypotheses.copy()
        else:
            df=hypotheses

        df=df.groupby(['src','tgt']).median().reset_index()
        df=df.pivot(index='src',columns='tgt',values='lomar')
        df=df.fillna(0)

        index = df.index.union(df.columns)
        df = df.reindex(index=index, columns=index, fill_value=0)

        df=df.sort_index()
        if square_mat:
            df.to_csv(filename.replace('.dot','sq.csv'))

        G = nx.from_pandas_adjacency(df,create_using=nx.DiGraph())

        from networkx.drawing.nx_agraph import write_dot
        write_dot(G,filename)

        return


    def getAlpha(self,dataframe_x_y_sigmay,N=500):
        """Carryout regression to estimate   \( \\alpha \). Given mean and variance of each y observation, we
           increase the number of pints by drawing N samples from a normal distribution of mean y and std dev sigma_y.
           The slope and p-value of a linear regression fit is returned

        Args:
          dataframe_x_y_sigmax (pandas DataFrame): columns x,y,sigmay
          N (int): number of samples drawn for each x to set up regression problem

        Returns:
          float,float: slope and p-value of fit

        """
        gf=pd.DataFrame(np.random.normal
                        (dataframe_x_y_sigmay.y,
                         dataframe_x_y_sigmay.sigmay,
                         [N,dataframe_x_y_sigmay.index.size]))

        RES=[dataframe_x_y_sigmay[['y','x']]]
        for i in gf.columns:
            xf=pd.DataFrame(gf.iloc[:,i])
            xf.columns=['y']
            xf['x']=dataframe_x_y_sigmay.x[i]
            RES.append(xf)
        RES=pd.concat(RES).dropna()

        lr=stats.linregress(RES.x,RES.y)
        return lr.slope,lr.pvalue


    def get_vector_from_dict(self,str_alph_val):
        """Calculate a probability distribution from string representation of
           labels : value read from decision tree models

        """
        vec_alph_val=str_alph_val.split(':')

        dict_label_float={}

        for x in vec_alph_val:
            if '.' not in x:
                dict_label_float[x] = float(vec_alph_val[vec_alph_val.index(x) +1].split()[0])
            else:
                d_k = x.split()
                df_k = ''
                if len(d_k) > 1:
                    for i in d_k[1:]:
                        df_k += i
                    dict_label_float[df_k] = float(vec_alph_val[vec_alph_val.index(x) +1].split()[0])

        prob_dist = np.zeros(len(dict_label_float))
        index = 0
        for i in dict_label_float:
            prob_dist[index] = dict_label_float[i]
            index += 1

        return prob_dist/prob_dist.sum()


    def trim_hypothesis(self,alternate_hypothesis_dataframe):
        """Compate current hypothesis dataframe with alternate_hypothesis_dataframe

        Args:
          alternate_hypothesis_dataframe ([pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)): alternate dataframe

        Returns:
          [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html): manipulated dataframe

        """
        df=self.hypotheses.copy()

        #df.set_index(['src','tgt']).merge


        return df