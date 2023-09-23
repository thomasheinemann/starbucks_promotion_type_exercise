
# base packages
import numpy as np
import pandas as pd

# sci-kit learn libraries
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline




"""The following transformer acts as a value set simplifyer of a dataframe X carrying information about a customers profile """
class simplify_feature_values(TransformerMixin):
    def fit(self, X,y=None):
        """fit function (dummy)"""
        return self

    def transform(self,X,y=None):
        """transform function for simplifying the value set of customer profile data contined in dataframe X """
        X_tmp=X.copy()

        # simplify became_member_on to its year
        X_tmp["became_member_on"]=(X["became_member_on"]/10000).astype(int)

        # simplify income to 10 k dollar steps
        X_tmp["income"]=(X["income"]/10000).astype(int)*10000

        # simplify age to 10 year steps
        X_tmp["age"]=(X["age"]/10).astype(int)*10

        return X_tmp



# the following transformer adds viewing preference attributes - for fitting model "build_viewing_prob_model" is required
class add_viewing_pref_features(TransformerMixin):
    def __init__(self, offer_viewing_threshold):
        """constructor function"""
        # initialize ML-model for viewing prob. prediction
        self.model=MultiOutputClassifier(LogisticRegression(random_state=0, C=1, max_iter=5000, class_weight="balanced" ))

        # call parameter settings procedure
        self.set_params(offer_viewing_threshold)
        return None

    def set_params(self,offer_viewing_threshold):
        """parameters are set"""
        self.offer_viewing_threshold=offer_viewing_threshold

        return self

    def fit(self, X, y=None):
        """fit function """
                
        tmp=X[X.offer_id !="no offer"] # dataframe X but constrained to offers 
        viewed_within_certain_time=(tmp["offer_viewed_time"]-tmp["offer_received_time"]<=self.offer_viewing_threshold)
        viewed_after_certain_time=(tmp["offer_viewed_time"]-tmp["offer_received_time"]>self.offer_viewing_threshold)  

        y=pd.DataFrame()
        y["view_pref"]=viewed_within_certain_time+viewed_after_certain_time
        y["view_pref2"]=viewed_within_certain_time+2*viewed_after_certain_time
        
        self.model.fit(tmp[["gender_M", "gender_F", "gender_O", "gender_nan", "age", "became_member_on", "income"]],y)# , sample_weight=tmp.duration)

        return self

    def transform(self,X,y=None):
        """ Transform function """

        pred_proba=self.model.predict_proba(X[["gender_M", "gender_F", "gender_O", "gender_nan", "age", "became_member_on", "income"]])
        viewing_prob=(pred_proba[0].T[1]) 

        X2=pd.DataFrame()  
        X2["early_viewing_pref"]= (pred_proba[1].T[1]>=pred_proba[1].T[2])  & (viewing_prob>=0.5)  
        X2["late_viewing_pref"]=  (pred_proba[1].T[1]<pred_proba[1].T[2])  & (viewing_prob>=0.5)   
        X2["early_viewing_prob"]=pred_proba[1].T[1]

        X2.index=X.index.copy()

        return pd.concat([X,X2], axis=1)


# Transformer selecting rows that belong to a specific offer of dataframe X carrying customer_profile and offer-related information  
class select_specific_offer_and_no_offer(TransformerMixin):
    def __init__(self, offer_id=None):
        """constructor function"""
        self.set_params(offer_id)
        return None

    def set_params(self,offer_id):
        """parameters are set"""
        self.offer_id=offer_id
        return self
    
    def fit(self, X,y=None):
        """fit function (dummy)"""
        return self

    def transform(self,X,y=None):
        """Transform function selecting rows of dataframe X referring to a specific offer saved in self.offer_id"""
        df_for_spec_offer=X[(X.offer_id==self.offer_id) | (X.offer_id==self.offer_id+"no offer") | (X.offer_id=="no offer")]
        return df_for_spec_offer

# the following transformer class adds information to whether the customers in their no-offer times could have been completed a specific offer or not
class adjust_completion_feature(TransformerMixin):
    def __init__(self, offer_portfolio):
        """constructor function"""
        self.set_params(offer_portfolio)
        return None

    def set_params(self,offer_portfolio):
        """parameters are set"""
        self.offer_portfolio=offer_portfolio
        return self

    def fit(self, X,y=None):
        """fit function (dummy)"""
        return self

    def transform(self,X,y=None):
        """Transform function leading to set the "completion" attribute for "no-offer offers" and informational offers"""
        

        difficulty=self.offer_portfolio.difficulty # difficulty of considered offer type set in "set_params"
        offer_id=self.offer_portfolio.id           # offer_id of considered offer type set in "set_params"
        
        # adjust "completion" attribute when informational offer types are considered
        if difficulty==0:
            # 1. step) adjust "completion" attribute for all entries of the current informational offer
            # left-join all rows for each person covering the considered offer with their no-offer rows to the right
            tmp=pd.merge(X[X.offer_id==offer_id], X[(X.offer_id=="no offer") | (X.offer_id==offer_id+"no offer")][["person","consumption","duration"]], on="person",how="left")
            A=tmp[["consumption_x"]]                                                    # grab consumptions column for all data sets of the considered offer type
            B=tmp[["consumption_y"]]                                                    # grab consumptions column for all data sets of the no offer
            C=tmp[["duration_x"]]                                                       # grab duration     column for all data sets of the considered offer type
            D=tmp[["duration_y"]]                                                       # grab duration     column for all data sets of the no offer
            # for each row in "tmp" set completed as: completed:=(consumption_offer * duration_nooffer >= consumption_nooffer * duration_offer)
            tmp.loc[tmp.offer_id==offer_id,"completed"]=np.array(list(map(lambda x : 1 if x==True else 0, A.values*D.values>=C.values*B.values )))
            tmp.rename(columns={"consumption_x" : "consumption"}, inplace=True)
            tmp.rename(columns={"duration_x" : "duration"}, inplace=True)
            tmp.drop(columns=["consumption_y","duration_y"], inplace=True)
            
            # 2. step) adjust "completion" attribute for all no-offer entries w.r.t. the current informational offer
            # right-join all rows for each person covering the considered offer with their no-offer rows to the right
            # tmp2=pd.merge(X[X.offer_id==offer_id][["person","consumption","duration"]], X[(X.offer_id=="no offer") | (X.offer_id==offer_id+"no offer")], on="person",how="right")
            tmp2=pd.merge(X[X.offer_id==offer_id][["person","consumption","duration"]].groupby(by=["person"]).sum(), X[(X.offer_id=="no offer") | (X.offer_id==offer_id+"no offer")], on="person",how="right")
            A=tmp2[["consumption_x"]]                                                   # grab consumptions column for all data sets of the considered offer type
            B=tmp2[["consumption_y"]]                                                   # grab consumptions column for all data sets of the no offer
            C=tmp2[["duration_x"]]                                                      # grab duration     column for all data sets of the considered offer type
            D=tmp2[["duration_y"]]                                                      # grab duration     column for all data sets of the no offer
            # for each row in "tmp2" set completed as: completed:=(consumption_offer * duration_nooffer < consumption_nooffer * duration_offer)
            tmp2.loc[(tmp2.offer_id=="no offer") | (tmp2.offer_id==offer_id+"no offer"),"completed"]=np.array(list(map(lambda x : 1 if x==True else 0, A.values*D.values<C.values*B.values )))
            tmp2.rename(columns={"consumption_y" : "consumption"}, inplace=True)
            tmp2.rename(columns={"duration_y" : "duration"}, inplace=True)
            tmp2.drop(columns=["consumption_x","duration_x"], inplace=True)

            # union tmp and tmp2 into dataframe X (has same attributes than the original one)
            X=pd.concat([tmp,tmp2])

        # adjust "completion" attribute when non-informational offer types (BOGO and discount) are considered
        else: 
            duration=24*self.offer_portfolio.duration
            X.loc[X.offer_id=="no offer","completed"]=np.array(list(map(lambda x : 1 if x==True else 0, (duration+1)*X[X.offer_id=="no offer"]["consumption"].values>difficulty* (X[X.offer_id=="no offer"]["duration"].values)  )))
        
        # rename "no offer"-offer_id into offer_id=offer_id + "no offer"   -> reason: "no offer"-offer_ids should bear different completion attributes w.r.t. different offer types
        X.loc[X.offer_id=="no offer","offer_id"]=[offer_id + "no offer"]*(X.loc[X.offer_id=="no offer"]).shape[0]
        return X


def starbucks_metrics(df,nir_correction_term):
    """
    metrics function
    
    INPUT
    df - dataframe with at least the columns "promoted" and "viewed"

    OUTPUT
    irr - incremental response rate (see formular or documentation)
    nir - net incremental reveniew (see formular or documentation)
    """
    # # calculate irr metric
    n_treat       = df.loc[df["promoted"] == 1,:].shape[0]   # number of treated (promoted) customers
    n_control     = df.loc[df["promoted"] == 0,:].shape[0]    # number of non-treated customers
    n_treat_purch = df.loc[df["promoted"] == 1, 'completed'].sum() # number of treated customers who purchased
    n_ctrl_purch  = df.loc[df["promoted"] == 0, 'completed'].sum()  # number of non-treated customers who purchased
    
    irr = -10000 if (n_treat==0) else 1 if  (n_control==0) else n_treat_purch / n_treat - n_ctrl_purch / n_control # metrics - incremental response rate w.r.t. viewing-(-5 is just a negative default)


    # calculate nir metric (shifted version)
    cons_treat_purch=(df.loc[ (df["promoted"] == 1) & (df["completed"] == 1),"consumption"]/df.loc[(df["promoted"] == 1) & (df["completed"] == 1),"duration"]).sum()
    cons_ctrl_purch=(df.loc[ (df["promoted"] == 0) & (df["completed"] == 1),"consumption"]/df.loc[(df["promoted"] == 0) & (df["completed"] == 1),"duration"]).sum()
    # cons_treat_purch=(df.loc[ (df["completed"] == 1)& (df["promoted"] == 0),"consumption"]/df.loc[(df["completed"] == 1)& (df["promoted"] == 0),"duration"]).sum()
    # cons_ctrl_purch=(df.loc[(df["completed"] == 0)& (df["promoted"] == 0),"consumption"]/df.loc[(df["completed"] == 0)& (df["promoted"] == 0),"duration"]).sum()

    nir=(cons_treat_purch-cons_ctrl_purch)-nir_correction_term*df.shape[0]

    return (irr, nir)
    
def score_prom(X,nir_correction_term):
    """scoring function - it is a essentially a product of the metrics irr and nir"""   
    
    score_df=X  # consider only those people that would be promoted with our strategy
    
    irr, nir = starbucks_metrics(score_df,nir_correction_term) # calcukate two different metrics

    main_score=abs(irr)*abs(nir)*np.sign(np.sign(irr)+np.sign(nir)-0.5)  # product of the metrics irr and nir  with a negative sign if both are negative

    return main_score,irr,nir

# the following classifier determines if a customer may need a promotion offer or not
class promotion_classifier(BaseEstimator):
    def __init__(self, prob_threshold=[0.5,0.5,0.5,0.5],estimator=[LogisticRegression(),LogisticRegression(),LogisticRegression()], features=["gender_M", "gender_F", "gender_O", "gender_nan", "age", "became_member_on", "income", "early_viewing_pref","late_viewing_pref","viewing_prob"]): # , "not_viewing_prob"]):
        """constructor function"""
        self.set_params(prob_threshold, estimator, features)
        return None

    def set_params(self,prob_threshold,  estimator, features):
        """parameters are set"""
        self.prob_threshold=prob_threshold

        self.estimator = estimator
        self.features=features

        return self
        
    def combine_probabilities_into_output(self,prob1,prob2,prob3,prob4): #,prob5):
        """
        Function aiming to combine three preditor probabilities into one
        
        INPUT
        prob1 - probability of event 1
        prob2 - probability of event 2
        prob3 - probability of event 3
        
        OUTPUT
        "YES" or "NO" depending on the combined condition
        """

        if (  (prob1>=self.prob_threshold2[0]) and (prob2>=self.prob_threshold2[1] ) and (prob3>=self.prob_threshold2[2])  and (prob4>=self.prob_threshold2[3]) ):
            return "Yes"
        else:
            return "No"
        
        return None

    
    def score(self, X, y):
        """scoring function - it is a essentially a product of the metrics irr and nir"""   
        y_pred=self.predict(X)# [self.features])
        score_df=X.iloc[np.where(y_pred=="Yes")]  # consider only those people that would be promoted with our strategy

        irr, nir = starbucks_metrics(score_df,self.nir_correction_term) # calcukate two different metrics
        
        if score_df.shape[0]<100:
            irr=-1000
            nir=-1000
        main_score=abs(irr)*abs(nir)*np.sign(np.sign(irr)+np.sign(nir)-0.5)  # product of the metrics irr and nir  with a negative sign if both are negative

        return main_score
    

    def fit(self,X,y=None):
        """fit procedure covering all classifiers for each target"""

        # build subsets among customers profiles depending on their CER states (viewed & completed)
        # train_data_promoted_buyer=(X[(X.completed==1) & (X.viewed==1)]).copy()        # type I
        # train_data_sure_buyer=(X[(X.completed==1) & (X.viewed==0) & (X.promoted==1)]).copy()             # type II
        # train_data_sure_not_buyer=(X[(X.completed==0) & (X.viewed==1)]).copy()          # type III
        # train_data_unpromoted_not_buyer=(X[(X.completed==0) & (X.viewed==0) & (X.promoted==1)]).copy()    # type IV
        train_data_promoted_buyer=(X[(X.completed==1) & (X.viewed==1)]).copy()        # type I
        train_data_sure_buyer=(X[(X.completed==1) & (X.viewed==0)]).copy()             # type II
        train_data_sure_not_buyer=(X[(X.completed==0) & (X.viewed==1)]).copy()          # type III
        train_data_unpromoted_not_buyer=(X[(X.completed==0) & (X.viewed==0) ]).copy()    # type IV


        # attach a new attribute "parameter" which is in if a person is of 
        train_data_sure_buyer['parameter']=0
        train_data_unpromoted_not_buyer['parameter']=1
        train_data_sure_not_buyer['parameter']=0
        train_data_promoted_buyer['parameter']=1

        
        # create 2 more subsets in which those who do not need a promotion have a 'parameter' value of zero and those who need or perhaps (we do not know) may not need a promotion get value 1.
        # these sets train the two main estimators
        temp1=pd.concat([train_data_sure_buyer, train_data_unpromoted_not_buyer])
        temp2=pd.concat([train_data_sure_not_buyer, train_data_promoted_buyer])

        # union upper subsets for trainingthe third estimator
        temp3=pd.concat([train_data_sure_buyer,train_data_sure_not_buyer, train_data_unpromoted_not_buyer,train_data_promoted_buyer])

        
        # build train and test set for first estimator  (II vs. IV) - see documentation
        X_train1=temp1[self.features]
        y_train1=np.array(temp1[['parameter']].values).ravel()

        # build train and test set for second estimator (I vs. III) - see documentation
        X_train2=temp2[self.features]
        y_train2=np.array(temp2[['parameter']].values).ravel()
        
        # build train and test set for third estimator ((I or IV) vs. (II or III)) - see documentation
        X_train3=temp3[self.features]
        y_train3=np.array(temp3[['parameter']].values).ravel()
        

        # fit first (sub)predictor
        if all(y_train1==y_train1[0]): 
            self.oneclassonly1= True
            self.oneclassvalue1= y_train1[0]
        else:
            self.oneclassonly1= False
            self.estimator[0].fit(X_train1,y_train1,sample_weight=temp1.duration)     # fit first estimator

        # fit second (sub)predictor
        if all(y_train2==y_train2[0]): 
            self.oneclassonly2= True
            self.oneclassvalue2= y_train2[0]
        else:
            self.oneclassonly2= False
            self.estimator[1].fit(X_train2,y_train2,sample_weight=temp2.duration)     # fit second estimator

        # fit third (sub)predictor
        if all(y_train3==y_train3[0]): 
            self.oneclassonly3= True
            self.oneclassvalue3= y_train3[0]
        else:
            self.oneclassonly3= False
            self.estimator[2].fit(X_train3,y_train3,sample_weight=temp3.duration)     # fit third estimator

        X2=X# [X.promoted==1]
        y_pred1,y_pred2,y_pred3=self.predictor_probs(X2)
        self.prob_threshold2=self.prob_threshold
        
        
        self.prob_threshold2[0]=np.percentile(y_pred1,self.prob_threshold[0]*100)
        self.prob_threshold2[1]=np.percentile(y_pred2,self.prob_threshold[1]*100)
        self.prob_threshold2[2]=np.percentile(y_pred3,self.prob_threshold[2]*100)
        self.prob_threshold2[3]=np.percentile(X2.early_viewing_prob.values,self.prob_threshold[3]*100)
        
        fitsize=X.shape[0]
        cons=(X.loc[(X.promoted==1)&(X.completed==1), "consumption"]/X.loc[(X.promoted==1)&(X.completed==1), "duration"]).sum()
        cons_ctrl=(X.loc[(X.promoted==0)&(X.completed==1), "consumption"]/X.loc[(X.promoted==0)&(X.completed==1), "duration"]).sum()
        # cons=(X.loc[(X.promoted==0)&(X.completed==1), "consumption"]/X.loc[(X.promoted==0)&(X.completed==1), "duration"]).sum()
        # cons_ctrl=(X.loc[(X.promoted==0)&(X.completed==0), "consumption"]/X.loc[(X.promoted==0)&(X.completed==0), "duration"]).sum()
        self.nir_correction_term=(cons-cons_ctrl)/fitsize
        print("nir_correction_term: ", self.nir_correction_term)

        return None
        
        

    def transform(self,X,y=None):
        """transform function (dummy)"""

        return X     

    def predictor_probs(self, X):

        """
        predict-funtion for the only one target value
        
        INPUT
        X - dataframe containing only customer data with only the attributes self.features

        OUTPUT
        array of values "Yes" or "No" that determines if each if the person provided will get a promotion according to the strategy we considered
        
        """

        # evaluate first (sub)predictor
        if self.oneclassonly1:
            y_pred1 =[self.oneclassvalue1]*X.shape[0]
        else:
            y_pred1 = self.estimator[0].predict_proba(X[self.features]).T[1]
        
        # evaluate second (sub)predictor
        if self.oneclassonly2:
            y_pred2 =[self.oneclassvalue2]*X.shape[0]
        else:
            y_pred2 = self.estimator[1].predict_proba(X[self.features]).T[1]
        
        # evaluate third (sub)predictor
        if self.oneclassonly3:
            y_pred3 =[self.oneclassvalue3]*X.shape[0]
        else:
            y_pred3 = self.estimator[2].predict_proba(X[self.features]).T[1]

        return y_pred1,y_pred2,y_pred3
    
    def predict(self, X):
        """
        predict-funtion for the only one target value
        
        INPUT
        X - dataframe containing only customer data with only the attributes self.features

        OUTPUT
        array of values "Yes" or "No" that determines if each if the person provided will get a promotion according to the strategy we considered
        
        """
        y_pred1,y_pred2,y_pred3=self.predictor_probs(X)
        

        return np.array(list(map(self.combine_probabilities_into_output,y_pred1,y_pred2,y_pred3,X.early_viewing_prob.values)))# ,X.viewing_prob.values)))
        
