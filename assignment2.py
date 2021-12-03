#################################
# Your name: Ido Borenstein
#################################

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import sample
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        sample=np.ndarray(shape=(m,2))
        s = np.sort(np.random.uniform(0,1,m))
        for i,x in enumerate(s):
            r=np.random.random()
            if (x>=0 and x<=0.2) or (x>=0.4 and x<=0.6) or x>=0.8:
                if r<=0.8:
                    y=1
                else:
                    y=0
            else:
                if r<=0.1:
                    y=1
                else:
                    y=0
            sample[i][0],sample[i][1]=x,y

        return sample



    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        true_errors=[]
        empirical_errors=[]
        m_s=[ m for m in range(m_first,m_last+1,step)]
        output=np.ndarray(shape=(len(m_s),2))
        for i,m in enumerate(m_s):
            empirical_error=0
            true_error=0
            for j in range(T):
                sample=self.sample_from_D(m)
                xs=sample[:,0]
                ys=sample[:,1]
                results=intervals.find_best_interval(xs,ys,k)
                empirical_error+=results[1]/m
                true_error+=self.calculate_true_error(results[0])
            true_errors.append(true_error/T)
            empirical_errors.append(empirical_error/T)
            output[i][0],output[i][1]=empirical_error,true_error
        plt.xlabel("m")
        plt.ylabel("error")
        plt.plot(m_s,empirical_errors,label="Empirical error")
        plt.plot(m_s,true_errors,label="True error")
        plt.show()
        return output

        
    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        true_errors=[]
        empirical_errors=[]
        k_s=[ m for m in range(k_first,k_last+1,step)]
        sample=self.sample_from_D(m)
        xs=sample[:,0]
        ys=sample[:,1]
        for k in range(k_first,k_last+1,step):
            results=intervals.find_best_interval(xs,ys,k)
            empirical_error=results[1]/m
            true_error=self.calculate_true_error(results[0])
            true_errors.append(true_error)
            empirical_errors.append(empirical_error)

        plt.xlabel("k")
        plt.ylabel("error")
        plt.plot(k_s,empirical_errors,label="Empirical error")
        plt.plot(k_s,true_errors,label="True error")
        plt.show()
        return k_s[empirical_errors.index(min(empirical_errors))]



    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        true_errors = []
        empirical_errors = []
        penalties = []
        penalties_empirical = []
        k_s=[ m for m in range(k_first,k_last+1,step)]
        sample=self.sample_from_D(m)
        xs=sample[:,0]
        ys=sample[:,1]
        for k in range(k_first,k_last+1,step):
            results=intervals.find_best_interval(xs,ys,k)
            empirical_error=results[1]/m
            true_error=self.calculate_true_error(results[0])
            penalty = self.penalty_function(2*k,0.1,m)
            penalties.append(penalty)
            penalties_empirical.append(penalty+empirical_error)
            true_errors.append(true_error)
            empirical_errors.append(empirical_error)


        plt.xlabel("k")
        plt.ylabel("value")
        plt.plot(k_s,empirical_errors,label="Empirical error")
        plt.plot(k_s,true_errors,label="True error")
        plt.plot(k_s,penalties,label="Penalty")
        plt.plot(k_s,penalties_empirical,label="Penalty + empirical")
        plt.show()
        return k_s[penalties.index(min(penalties))]



    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        sample = self.sample_from_D(m)
        np.random.shuffle(sample)
        size = int(m/5)
        sample_validation = sample[:size]
        sample_training = sample[size:]
        sample_training = np.array(sorted(sample_training,key=lambda x:x[0]))
        sample_validation = np.array(sorted(sample_validation,key=lambda x:x[0]))
        validation_x,validation_y,training_x,training_y= sample_validation[:,0],sample_validation[:,1],sample_training[:,0],sample_training[:,1]
        validation_errors = []
        for k in range(1,11):
            results = intervals.find_best_interval(training_x,training_y,k)
            inters = results[0]
            validation =  self.validation_error(inters,validation_x,validation_y)
            validation_errors.append(validation)
        return np.argmin(validation_errors)+1
            



    #################################
    # Place for additional methods
    def validation_error(self,inters,xs,ys):
        error=0
        for i,x in enumerate(xs):
            if ys[i]==1:
                inside=False
                for inter in inters:
                    if x >=inter[0] and x<=inter[1]:
                        inside=True
                        break
                if not inside:
                    error+=1
            else:
                for inter in inters:
                    if x >=inter[0] and x<=inter[1]:
                        error+=1
                        break
           
        return error/len(xs)



    
    def calculate_true_error(self,inters):
       
        A = [(0,0.2),(0.4,0.6),(0.8,1)]
        B= [(0.2,0.4),(0.6,0.8)]
        # first check the case in which x is in A:
        in_A=0
        for cut in A:
            for inter in inters:
                in_A+=self.claculate_matchs(inter,cut)
        #x is in B:
        in_B=0
        for cut in B:
            for inter in inters:
                in_B+=self.claculate_matchs(inter,cut)
        return in_A*0.2+in_B*0.9+(0.6-in_A)*0.8+(0.4-in_B)*0.1
    
    def claculate_matchs(self,inter,cut):
        #the cut is inside  the inter:
        if cut[0]>=inter[0] and cut[1]<=inter[1]:
            return cut[1]-cut[0]
        #the inter is inside the cut:
        if inter[0]>=cut[0] and inter[1]<=cut[1]:
            return inter[1]-inter[0]
        #inter starts after cut but has intersction:
        if inter[0]>=cut[0] and cut[1] >=inter[0] and inter[1]>=cut[1]:
            return cut[1]-inter[0]
        #cut starts after inter but has intersction:
        if cut[0]>=inter[0] and inter[1] >=cut[0] and cut[1]>=inter[1]:
            return inter[1]-cut[0]
        
        return 0


    def penalty_function(self,vc_dim,delta,n):
        return 2*np.sqrt((np.log(2/delta) + vc_dim)/n)
        
    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)

