import pygraphviz as pgv
import networkx as nx
import numpy as np
import re
import os
from scipy import stats
import warnings
import pandas as pd
import glob
from tqdm import tqdm
warnings.filterwarnings('ignore')

class Hypothesis(object):

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
                variable_bin_map[param] = np.array([-4, 4])
            # Parameters with strong indications of frequency or opinion, but no explicit
            # 'strongly agree' / 'strongly disagree'. Examples responses
            #  - 'never'/'always', 'very good', 'very bad' 
            if param in ['abdefctw',"abpoor","bible", "godchnge", "intmil", "pray", "prayfreq", "viruses"]:
                variable_bin_map[param] = np.array([-3, 3])
            # Parameters with clear positioning, but no indication of intensity. Ex: "not fired", "fired" 
            elif param in ['colcom',"colmil", "conlabor", "grass", 'libcom','libmil','libhomo',
            'libmslm', 'spkcom','spkmil','taxrich']:
                variable_bin_map[param] = np.array([-2, 2])
            # Yes / No, Support / Don't support options
            else:
                variable_bin_map[param] = np.array([-1, 1])
        
        self.NMAP = variable_bin_map

        # Using manually encoded labels, since there is no quantizer for GSS data
        # In qbiome data, this can be obtained from quantizer.labels 
        # Each element in self.NMAP only has two options, so I will include two
        # choices for each parameter, for now.

        labels = {'no': 0}
        self.LABELS = labels

        self.gsss_interval = [x for x in self.NMAP.keys()]

        self.gsss = list(set(self.gsss_interval))

        self.total_samples = total_samples
        self.detailed_labels = detailed_labels

        self.decision_tree = None
        self.tree_labels = None
        self.tree_edgelabels = None
        self.TGT = None
        self.SRC = None
        self.no_self_loops = no_self_loops
        self.hypotheses=pd.DataFrame(columns=['src','tgt','lomar','pvalue'])

    def regularize_distribution(self,prob,l,e=0.005):
        """Regularize probability distribution
           using exponential decay to map non-detailed output of a
           single maximum likelihood label to a probability distribution.
           Used when detailed output is not available.

        Args:
          prob (float): probability of single output label of type str
          e (float, optional): small value to regularize return probability of 1.0 (Default value = 0.005)
          l (str): output label

        Returns:
          numpy.ndarray: probability distribution

        """

        labels=np.array(list(self.LABELS.keys()))
        yy=np.ones(len(labels))*((1-prob-e)/(len(labels)-1))
        yy[np.where(labels==l)[0][0]]=prob-e
        dy=pd.DataFrame(yy).ewm(alpha=.8).mean()
        dy=dy/dy.sum()
        return dy.values


    def createTargetList(self):
        """Create list of decision trees

        Returns:
          list[str]: list of paths to decision tree models in qnet

        """

        if self.model_path is not None:
            decision_trees = glob.glob(
                self.model_path+'*.dot')
            decision_trees_ = [x for x in decision_trees]
        else:
            raise Exception('self.model_path is not set')
        return decision_trees_

    def deQuantize_lowlevel(self,
                    key,
                    bin_arr):
        """Low level dequantizer function

        Args:
          key (str): quantized level (str) or nan
          bin_arr (numpy.ndarray): 1D array of floats

        Returns:
          float: dequantized value

        """

        if key is np.nan or key == 'nan':
            return np.nan
        lo = self.LABELS[key]
        hi = lo + 1
        val = (bin_arr[lo] + bin_arr[hi]) / 2
        return val

    def deQuantizer(self,
                    key,
                    gss_prefix):
        """Dequantizer function calling low level deQuantize_lowlevel to 
        operate with incomplete gss names.

        Args:
          key (str): quantized level (str) or nan
          gss_prefix (str): prefix of gss name

        Returns:
          float: median of dequantized value

        """
        vals = []
        # average over all
        for gss_key in self.NMAP:
            if gss_prefix in gss_key:
                vals.append(
                    self.deQuantize_lowlevel(key,
                                        self.NMAP[gss_key]))

        return np.median(vals)

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
        bin_name=self.TGT
        gss_prefix = ''

        # ----------------------------------------
        # Q is 1D array of dequantized values
        # corresponding to levels for TGT
        # ----------------------------------------
        Q=np.array([self.deQuantizer(
            str(x).strip(),
            gss_prefix)
                    for x in self.LABELS.keys()]).reshape(
                            len(self.LABELS),1)

        mux=0
        varx=0
        for k in Probability_distribution_dict:
            p = Probability_distribution_dict[k]

            mu_k=np.dot(p.transpose(),Q)
            var_k=np.dot(p.transpose(),(Q*Q))-mu_k*mu_k

            mux = mux + Sample_fraction[k]*mu_k
            varx = varx + Sample_fraction[k]*var_k
        return mux,np.sqrt(varx/self.total_samples)
    
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
        # is a dict mapping leafnode id to output label key
        #
        # frac and prob are sample fraction and probability of
        # output label in leaf node, parsed from dotfile
        #
        # SUM is the total sample fraction captured by nodeset
        cLeaf=[x for x in nodeset
               if self.decision_tree.out_degree(x)==0
               and self.decision_tree.in_degree(x)==1]

        oLabels={k:v.split('\n')[0]
                 for (k,v) in self.tree_labels.items()
                 if k in cLeaf}

        frac={k:float(v.split('\n')[2].replace('Frac:',''))
              for (k,v) in self.tree_labels.items()
              if k in cLeaf}
              
        if not self.detailed_labels:
            prob={k:float(v.split(oLabels[k]+':')[1].split(' ')[0])
                  for (k,v) in self.tree_labels.items()
                  if k in cLeaf}

            ## Get a kernel based distribution here.
            # self.alphabet=['A',...,'E']
            # prob is regularize_distribution to get a dict {nodeid: [p1,..,pm]}
            
            prob__={k:self.regularize_distribution(prob[k],oLabels[k])
                    for k in prob}
            prob=prob__
        else:
            prob={k:self.get_vector_from_dict(v.split(oLabels[k]+':')[1].split(' ')[0])
                  for (k,v) in self.tree_labels.items()
                  if k in cLeaf}

        SUM=np.array(frac.values()).sum()

        ## mean and sample estimate of standard deviation
        mu_X,sigma_X=self.getNumeric_at_leaf(prob,frac)
        return (mu_X,sigma_X),SUM

    
    def getNumeric_internal(self,
               dict_id_reached_by_edgelabel,
               bin_name):
        """Dequantize labels on graph non-leaf nodes

        Args:
          dict_id_reached_by_edgelabel (dict[int,list[str]]): dict mapping nodeid to array of keys with str type
          bin_name (str): gss name

        Returns:
          dict[int,float]: dict mapping nodeid to  dequantized values of float type

        """

        gss_prefix = ''

        R={}
        for k in dict_id_reached_by_edgelabel:
            v = dict_id_reached_by_edgelabel[k]
            R[k]=np.median(
                np.array([self.deQuantizer(
                    str(x).strip(),
                    gss_prefix) for x in v]))
        return R


    
    def getHypothesisSlice(self,nid):
        """Generating impact of node nid with source label prefix. Note that there can be multiple
           nodes in the tree with label that match with the source label prefix.

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
        
        # nextedge is dict: {nodeid nn:  key array by which child nn is reached}
        num_nextedge=self.getNumeric_internal(
            nextedge,
            bin_name
            =self.tree_labels[str(
                nid)])
        for (k,v) in edgeProp.items():
            num_nextedge[k]=np.append(
                num_nextedge[k],[v[0],v[1]])

        RF=pd.DataFrame(num_nextedge)
        RF.index=['x', 'y','sigmay']
        return RF

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

        decision_trees = self.createTargetList()

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

            ns_ = re.split(r'(.*)_(\d+)', self.TGT)
            self.hypotheses = self.hypotheses.append(
                {'src':self.SRC,
                 'tgt':''.join(ns_[:-2]),
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
        """Carry out regression to estimate   \( \\alpha \). Given mean and variance of each y observation, we
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
           alphabet : value read from decision tree models

        """
        vec_alph_val=str_alph_val.split()

        dict_label_float={}

        for x in vec_alph_val:
            y=x.split(':')
            dict_label_float[y[0]]=float(y[1])

        prob_dist = np.zeros(len(self.LABELS.keys()))
        for i in dict_label_float:
            prob_dist[self.LABELS[i]] = dict_label_float[i]

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