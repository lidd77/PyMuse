#!/usr/bin/env python
import argparse,random,os
import pandas as pd
import  numpy as np
from sklearn import preprocessing

__version__ = '1.0.4'
__pandas_version = '0.22.0'
__argparse_version = '1.1'
__numpy_version = '1.14.2'

class pymuse(object) :
    '''A wrapped  pymuse class'''

    def __init__(self):
        parser = argparse.ArgumentParser(prog="pymuse",description='this is a MUSE approach for identifying high-confidence candidate interacting proteins')
        parser.add_argument('--version', '-V', action='version', version='%(prog)s ' + __version__)
        parser.add_argument('--interaction','-i',required=True,dest="interaction",help="interaction.csv")
        parser.add_argument('--control','-c',required=True,dest="control",help="control.csv")
        parser.add_argument('--prey','-p',required=True,dest="prey",help="prey.csv")
        parser.add_argument('--crapome','-m',required=True,dest="crapome",help="CRAPome.csv")
        parser.add_argument('--outdir','-o',required=False,default=".",dest="outdir",help="output directory path[.]")
        parser.add_argument('--prefix','-x',required=False,default="tmp_",dest="prefix",help="a prefix name of the output file [tmp_]")
        parser.add_argument('--cvtype','-t',default=True,dest="cvtype",help="sim_cv : cv calculation method ,True for robust.[True]")
        parser.add_argument('--plist','-r',type=float,metavar='start end num',nargs='+',default=[0.02,1,50],dest="prob",help="sim_cv : start end step default: 0.02 1 0.02")
        parser.add_argument('--replicates','-e',type=int,metavar='replicate number',default=1000,dest="rep",help="sim_cv : replicate number default: 1000")
        self.args = parser.parse_args()

    def merge_data(self):
        '''to intergrated  three files from interaction.csv ,control.csv and prey.csv'''

        with open(self.args.interaction,'r') as interaction_f :
            with open(self.args.control,'r') as control_f :
                with open(self.args.prey,'r') as prey_f :
                    interaction_df,control_df,prey_df = [ pd.read_csv(file_handle,header=0) for file_handle in (interaction_f,control_f,prey_f)]
                    interaction_df['Control'] = False
                    control_df['Control'] = True
                    g = pd.concat([interaction_df,control_df],ignore_index=True,axis=0).pipe(pd.merge,prey_df,how='inner',on='Prey').groupby('Bait')
                    self.data_merge = pd.concat([ group.assign(Count = lambda x : x.Count/(sum(x.Count)*x.Length)) for name,group in g] ,ignore_index=True,axis=0)

    @classmethod
    def cv_calculation(cls,data_df,p,cvtype):
        nrow,ncol = data_df.shape
        # cv_tmp_df = data_df.apply(lambda  x : random.sample(x.tolist(),nrow),axis=0).apply(lambda x : x if np.sum(x) > 0 else [np.nan] * len(x),axis=1).dropna(axis=0,how='all').apply(lambda x: x / (sum(x**p) ** (1/p)),axis=1)
        cv_tmp_df = data_df.apply(lambda  x : random.sample(x.tolist(),nrow),axis=0).apply(lambda x : x if np.sum(x) > 0 else [np.nan] * ncol,axis=1).dropna(axis=0,how='all').apply(lambda x: x / (sum(x**p) ** (1/p)),axis=1)
        # cv_tmp_df = data_df.apply(lambda  x : random.sample(x.tolist(),nrow),axis=0).\
        #     apply(lambda x : x if np.sum(x) > 0 else [np.nan] * ncol,axis=1).\
        #     dropna(axis=0,how='all').\
        #     apply(lambda x: x / (sum(x**p) ** (1/p)),axis=1)
        # print(cv_tmp_df.shape)
        # print(cv_tmp_df)

        dratios = cv_tmp_df.values.flatten()[cv_tmp_df.values.flatten() > 0]

        if cvtype :
            #cv robust alternative calculation
            mad_in_r = np.median(np.abs(dratios-np.median(dratios)))  *  1.4826
            return  mad_in_r/np.median(dratios)
        else :
            return  np.std(dratios,dtype='float64',ddof=1) / np.mean(dratios)

    @classmethod
    def get_cv(cls,data_df,p,replicate_number,cvtype):
        rep_n = np.linspace(1,replicate_number,replicate_number)
        rep_cv_tmp_df = pd.DataFrame({'rep':rep_n})


        cv_rep_tmp = [ pymuse.cv_calculation(data_df,p,cvtype) for rep_each in rep_n]
        rep_cv_tmp_df = rep_cv_tmp_df.assign(cv=cv_rep_tmp)
        cv_value = rep_cv_tmp_df['cv'].median()

        return cv_value

    @classmethod
    def select_p(cls,cv_p_df):
        slopes = np.abs(np.diff(cv_p_df['med_cv']) / np.diff(cv_p_df['p']))
        return  cv_p_df['p'][np.argmax(slopes)]

    def sim_cv(self):
        '''to calculate CV and select p '''
        tmp_dat  = self.data_merge[['Bait','Prey','Count']]
        acast_data_matrix = pd.pivot_table(tmp_dat,values='Count',index=['Prey'],columns=['Bait'],aggfunc=sum,fill_value=0,dropna=False).pipe(np.matrix)
        mat_multipy_df =  pd.DataFrame(acast_data_matrix *  np.diag(1/pd.DataFrame(acast_data_matrix.sum(0)).loc[0]))
        p_list = np.linspace(self.args.prob[0],self.args.prob[1],self.args.prob[2])
        sim_cv_p_df = pd.DataFrame({'p':np.linspace(self.args.prob[0],self.args.prob[1],self.args.prob[2])})
        sim_cv_p_df = pd.DataFrame({'p':p_list})

        tmp_cv = [ pymuse.get_cv(mat_multipy_df,p,self.args.rep,self.args.cvtype) for p  in  p_list]
        sim_cv_p_df = sim_cv_p_df.assign(med_cv=tmp_cv)

        self.p_select_value =  pymuse.select_p(sim_cv_p_df)


        print(sim_cv_p_df)

    def cal_scores_ctrl(self,p):
        print('p %s ' % p)
        tmp_dat = self.data_merge[['Bait','Prey','Count','Control']]
        t1 = tmp_dat.query('Control == True').groupby(['Prey']).Count.agg(lambda x : sum(x ** p))
        ctrl_sums = pd.DataFrame({'Prey':t1.index,'S':t1.values})
        g  = tmp_dat.merge(ctrl_sums,how='left' ,on='Prey').fillna(0).query('Control == False').groupby(['Bait','Prey'])
        self.scores_ctrl_df = pd.DataFrame([ {'Bait':name[0],'Prey':name[1],'Score':sum(group.Count**p) ** (2/p) / (np.mean(group.S) + sum(group.Count**p)) ** (1/p) }  for name ,group in g ] ).sort_values(by='Score', ascending=False, na_position='first')
        print(self.scores_ctrl_df)

    def cal_scores_notrl(self,p):
        g_one = self.data_merge.query('Control == False')[['Bait','Prey','Count']].groupby(['Prey'])
        g = pd.concat([  gr.assign(D= lambda x : sum(x.Count ** p) ** (1/p)  )   for n,gr in g_one  ] ).groupby(['Bait','Prey'])
        self.noctrl_scores_df = pd.DataFrame([ {'Bait':name[0],'Prey':name[1] , 'nScore': sum(group.Count**p) ** (2/p)/ np.mean(group.D)} for name,group in g ] ).sort_values(by='nScore', ascending=False, na_position='first')
        print(self.noctrl_scores_df)

    def output_result(self):
        out_file = os.path.join(self.args.outdir,self.args.prefix + '.csv')
        with open(self.args.crapome,'r') as crap_db_f :
            crap_df = pd.read_csv(crap_db_f,header=0) #TODO  crapome database file is not be used  with  muse R source codes
            output_df = self.scores_ctrl_df.merge(self.noctrl_scores_df,how='left',on=['Bait','Prey'])
            print(output_df)
            min_max_scaler = preprocessing.MinMaxScaler()
            # min_max_scaler.fit_transform(output_df.Score.reshape(output_df.shape[0],1)).flatten()
            scale_score_df = output_df.assign(Score_scale=min_max_scaler.fit_transform(output_df.Score.reshape(output_df.shape[0],1)).flatten())
            scale_score_df.to_csv(out_file,index=False,columns=['Bait','Prey','Score','Score_scale','nScore'],header=True)

    def run(self):
        self.merge_data()
        self.sim_cv()
        self.cal_scores_ctrl(self.p_select_value)
        self.cal_scores_notrl(self.p_select_value)
        self.output_result()

if __name__ == '__main__' :
    pymuse().run()





