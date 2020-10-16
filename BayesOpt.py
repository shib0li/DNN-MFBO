import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
import os
from hdf5storage import loadmat
from hdf5storage import savemat

import ExpUtils as util
from MFidelModel import MFNN


class BayesOptMFMM:
    def __init__(self, config, res_path, TF_GPU_USAGE=1):
        
        self.config = config
        self.M = self.config['model']['SynData'].Nfid
        self.dim = self.config['model']['SynData'].dim
        self.logger = self.config['logger']
        self.res_path = res_path

        # Initialize mf model
        self.model = MFNN(self.config['model'], TF_GPU_USAGE=TF_GPU_USAGE) 
        self.Fstar, self.infer_optser, self.handler = self.model.train() 
        self.Wpost, self.Lpost = self.acquire_posterior()
        self.NNs = self.duplicate()
        
        lb = self.model.SynData.lb
        ub = self.model.SynData.ub
        
        self.Xquery = tf.Variable(tf.random.uniform(minval=lb, maxval=ub, shape=[1,self.dim]), dtype=util.tf_type)
        
        # initial features
        self.features = []
        self.outputs = []
        for m in range(self.M):
            feature, output = self.init_mf_model(self.Xquery, self.Wpost, m)
            self.features.append(feature)
            self.outputs.append(output)
            
        # standard cdf 
        self.standard_normal = tfp.distributions.Normal(loc=0.0, scale=1.0)
        
        # marginal by moment matching propagation
        self.alpha, self.v = self.propagation()

        # mutual infomation optimization
        self.neg_mutual_info = []
        self.mutual_info_optimizers = []
        for m in range(self.M):
            self.neg_mutual_info.append(-self.batch_eval_mutual_information(m))
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.neg_mutual_info[m], 
                                                                method='L-BFGS-B', 
                                                                var_to_bounds={self.Xquery: [self.model.SynData.lb, self.model.SynData.ub]},
                                                                var_list=[self.Xquery],
                                                                options = {'maxiter': 50000,
                                                                       'disp': None,
                                                                       'maxfun': 50000,
                                                                       'maxcor': 50,
                                                                       'maxls': 50,
                                                                       'eps':self.config['OptInfoStep'],
                                                                       'ftol' : 1.0 * np.finfo(float).eps})
            self.mutual_info_optimizers.append(optimizer)
       

        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=TF_GPU_USAGE)

        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
                gpu_options=gpu_options,
            )
        )
        
        self.sess.run(tf.compat.v1.global_variables_initializer())
        
    def acquire_posterior(self):

        Wpost = []
        Lpost = []
        for m in range(self.M):
            Km = self.config['model']['feature']['Klist'][m]
            Wm = tf.constant(self.handler.run(self.model.tf_W_list[m]))
            Lm = tf.constant(self.handler.run(self.model.tf_Wvar_Chol_list[m]))
            Wpost.append(Wm)
            Lpost.append(Lm)
        
        return Wpost, Lpost


    def duplicate(self):
        """ Ducpliate the learned surrogate model for the convinience of inference """
        weights_list = []
        biases_list = []
        layers_list = []
        activation = self.config['model']['feature']['activation']
        for m in range(self.M):
            weights = self.handler.run(self.model.mf_encode_list[m].weights)
            biases = self.handler.run(self.model.mf_encode_list[m].biases)
            weights_list.append(weights)
            biases_list.append(biases)
            layers_list.append(self.model.mf_encode_list[m].layers)
        
        NNs = []
        for m in range(self.M):
            nn = util.CopyEncodeNN(layers_list[m], activation, weights_list[m], biases_list[m])
            NNs.append(nn)
        # end for
        
        return NNs
    
    def init_mf_model(self, X, Wlist, m):
        if m == 0:
            fm = self.NNs[m].forward(X)
            augment_fm = tf.pad(fm, tf.constant([[0, 0,], [0, 1]]), constant_values=1.0)
            ym = tf.matmul(augment_fm, Wlist[m])
            
            return augment_fm, ym
        else:
            fm = self.NNs[0].forward(X)
            augment_fm = tf.pad(fm, tf.constant([[0, 0,], [0, 1]]), constant_values=1.0)
            prev = tf.matmul(augment_fm, Wlist[0])
            
            for i in range(1,m+1):
                augment_input = tf.concat([prev, X], axis=1)
                fm = self.NNs[i].forward(augment_input)
                augment_fm = tf.pad(fm, tf.constant([[0, 0,], [0, 1]]), constant_values=1.0)
                prev = tf.matmul(augment_fm, self.Wpost[i])
                
            return augment_fm, prev

    
    def eval_base_fid(self):
        """ marginal p(f_0(x) | D) """
        phi = self.features[0]
        Wm = self.Wpost[0]
        Lm = self.Lpost[0]
        
        am = tf.matmul(phi, Wm)
        vm = phi @ (Lm @ tf.transpose(Lm))
        vm = tf.reduce_sum(vm * phi, axis=1)
        
        return tf.squeeze(am), tf.squeeze(vm)
    
    def moment_match(self, alphax, vx, m):
        """
        p(f_{m-1} | D) => p(f_m(x) | D)
        
        Args:
            alphax: mean of the previous marginal
            vx: variance of the previous marginal
            m: next fidelity
        """
        
        def cond_dist(nodes):
            nodes = tf.reshape(nodes, [-1,1])
            Nq = self.config['NQuad']
            
            augment_input = tf.concat((nodes, tf.repeat(self.Xquery, Nq, axis=0)), axis=1)
            feature = self.NNs[m].forward(augment_input)
            augment_feautre = tf.pad(feature, tf.constant([[0, 0,], [0, 1]]), constant_values=1.0)
            
            mu = tf.squeeze(tf.matmul(augment_feautre, self.Wpost[m]))
            
            term1 = tf.reduce_sum(tf.square(tf.matmul(augment_feautre, self.Lpost[m])), axis=1)
            term2 = tf.square(tf.squeeze(tf.matmul(augment_feautre, self.Wpost[m])))
            
            return mu, term1+term2
        
        margin_mean, margin_square_mean = util.tf_quad(cond_dist, alphax, vx, self.config['NQuad'])
        margin_var = margin_square_mean - tf.square(margin_mean) 
        
        return margin_mean, margin_var
    
    
    def batch_moment_match(self, batch_alpha, batch_v, m):
        
        def cond_dist(nodes):
            nodes = tf.reshape(nodes, [-1,1])
            Nq = tf.size(nodes)
            
            augment_input = tf.concat((nodes, tf.repeat(self.Xquery, Nq, axis=0)), axis=1)
            feature = self.NNs[m].forward(augment_input)
            augment_feature = tf.pad(feature, tf.constant([[0, 0,], [0, 1]]), constant_values=1.0)
            
            mu = tf.matmul(augment_feature, self.Wpost[m])
            
            term1 = tf.reshape(tf.reduce_sum(tf.square(tf.matmul(augment_feature, self.Lpost[m])), axis=1), [-1,1])
            term2 = tf.square(tf.matmul(augment_feature, self.Wpost[m]))
            
            return mu, term1+term2
        
        margin_mean, margin_square_mean = util.batch_tf_quad(cond_dist, batch_alpha, batch_v, self.config['NQuad'])
        margin_var = margin_square_mean - tf.square(margin_mean)
        
        return margin_mean, margin_var
        

    def propagation(self):
        """ calculate [alpha_m, v_m], m = [0,M-1] """
        alpha = []
        v = []
        
        am, vm = self.eval_base_fid()
        alpha.append(am)
        v.append(vm)
        
        for m in range(1,self.M):
            am, vm = self.moment_match(alpha[-1], v[-1], m)
            alpha.append(am)
            v.append(vm)
        
        return alpha, v
    
    def cross_propagation(self, fm, m):
        """ return eta_M, gamma_M given observation fm at fidelity m"""
        # base case m+1
        acurr, vcurr = self.moment_match(fm, 0.0, m+1)
        
        # propagate to M
        for i in range(m+2, self.M):
            anext, vnext = self.moment_match(acurr, vcurr, i)
            acurr = anext
            vcurr = vnext

        return acurr, vcurr
    
    def batch_cross_propagation(self, batch_fm, m):

        Nquals = tf.size(batch_fm)
        var = tf.repeat([0.0], Nquals, axis=0)
        
        batch_acurr, batch_vcurr = self.batch_moment_match(batch_fm, var, m+1)
        
        for l in range(m+2, self.M):
            batch_anext, batch_vnext = self.batch_moment_match(batch_acurr, batch_vcurr, l)
            batch_acurr = batch_anext
            batch_vcurr = batch_vnext
        
        return batch_acurr, batch_vcurr

    def batch_eval_condition_entropy(self, m, Fstar):
        
        Fstar = tf.constant(Fstar, dtype=util.tf_type)
        [nodes, weights] = util.quadrl(self.config['NQuad'])  
        
        # transform nodes
        dc_nodes = nodes * tf.sqrt(self.v[m]) + self.alpha[m]
        # batch propagation given Fstar
        batch_eta, batch_gamma = self.batch_cross_propagation(nodes, m)
        
        gs = (Fstar - batch_eta) / tf.sqrt(batch_gamma)
        # broadcast: Nq by |Fstar| + weights: Nq by 1
        log_cdf_gs = self.standard_normal.log_cdf(gs) + tf.reshape(tf.log(weights), [-1,1])
        tr_log_cdf_gs = tf.transpose(log_cdf_gs)
        # broadcast: |Fstar| by Nq - Nq by 1
        coeff = tf.exp(tr_log_cdf_gs - tf.math.reduce_logsumexp(tr_log_cdf_gs, axis=1, keepdims=True))
        batch_fm_mean = tf.matmul(coeff, tf.reshape(dc_nodes,[-1,1]))
    
        # broadcast [1 by |Fstar|] - [Nq by 1] = [|Fstar| by Nq]
        diff = tf.reshape(dc_nodes, [1,-1]) - batch_fm_mean
        batch_fm_var = tf.reduce_sum(coeff*tf.square(diff), axis=1, keepdims=True)
        
        batch_cond_h = 0.5 * tf.math.log(2 * np.pi *np.e * batch_fm_var)
   
        return  batch_cond_h
        
    def batch_eval_mutual_information(self, m):
        if m < self.M - 1:
            average_cond_entropy = tf.reduce_mean(self.batch_eval_condition_entropy(m, self.Fstar))
            margin_entropy = 0.5 * tf.math.log(2*np.pi*np.e*self.v[m])
            
            return margin_entropy - average_cond_entropy
        else:
            Fstar = tf.constant(self.Fstar, dtype=util.tf_type)            
            
            gs = (Fstar - self.alpha[m]) / tf.sqrt(self.v[m])
            sigm = tf.sqrt(self.v[m])
            
            term1 = 0.5*tf.math.log(2*np.pi*np.e*self.v[m]) + self.standard_normal.log_cdf(gs)         
            term2 = gs*tf.exp(self.standard_normal.log_prob(gs) - tf.log(2.0) - self.standard_normal.log_cdf(gs))
            
            
            average_cond_entropy = tf.reduce_mean(term1 - term2)
            margin_entropy = 0.5 * tf.math.log(2 * np.pi *np.e * self.v[m])

            return margin_entropy - average_cond_entropy

    def optimize(self):


        hist_simple_opt = []  # inference regret
        hist_infer_opt = []  # simple regret
        hist_argm = []
        hist_argx = []
        hist_cost = []
        hist_Fstar = []
        
        init_pts = np.array(self.config['model']['SynData'].Ntrain_list)
        init_cost = np.sum(np.array(self.config['cost']) * init_pts)
        hist_cost.append(init_cost)
        
        simple_opt = -np.inf
        
        global_maximum = self.model.SynData.maximum
        
        print('Initialize BO on', self.config['model']['domain'])
        print('  - Global optimum =', global_maximum)
        print('  - Cost =', np.array(self.config['cost']))
        print('  - Initial cost =', init_cost)
        self.logger.write('Initialize BO on ' + self.config['model']['domain'] + '\n')
        self.logger.write('  - Global optimum = ' +  str(global_maximum) + '\n')
        self.logger.write('  - Cost = ' + np.array_str(np.array(self.config['cost'])) + '\n')
        self.logger.write('  - Init cost = ' + str(init_cost) + '\n')
        
        for t in range(self.config['maxiter']):
            
            t0 = time.time()
           
            argm, argx, info_gain = self.eval_mf_query()
            
            print(argx)
            yq, ystar, argx = self.model.SynData.append(argx, argm)
            print(argx)
            
            self.logger.write('    * debug = ' + str(yq) + '\n')
            self.logger.write('    * debug = ' + str(ystar) + '\n')
            
            hist_argm.append(argm)
            hist_argx.append(argx)
            hist_cost.append(self.config['cost'][argm])
            hist_Fstar.append(np.mean(np.array(self.Fstar)))

            if ystar >= simple_opt:
                simple_opt = ystar
            
            hist_simple_opt.append(simple_opt)
            self.infer_opt = self.model.SynData.query(self.infer_optser, self.M-1)
            
            hist_infer_opt.append(self.infer_opt)
            
            self.Fstar, self.infer_optser, self.handler = self.model.train()
            self.Wpost, self.Lpost = self.acquire_posterior()
            self.NNs = self.duplicate()
            
            
            
            t1 = time.time()
            print('iter #'+str(t))
            print('argmax:', argm, argx, 'gain:', info_gain, 'queried value:', yq, ystar)
            print('- simple optimum: ', simple_opt)
            print('- infere optimum: ', self.infer_opt)
            print('- spending so far: ', np.sum(np.array(hist_cost)))
            self.logger.write('  - iter #' + str(t) + '\n')
            self.logger.write('    * simple optimum = ' + str(simple_opt) + '\n')
            self.logger.write('    * infere optimum = ' + str(self.infer_opt) + '\n')
            self.logger.write('    * Fstar = ' + str(self.Fstar) + '\n')
            self.logger.write('    * mean Fstar = ' + str(np.mean(np.array(self.Fstar))) + '\n')
            self.logger.write('    * argmax: ' +  'm = ' + str(argm) + ' argx= ' + np.array_str(argx) +\
                              ' info gain = ' + np.array_str(info_gain) + '\n')
            self.logger.write('    * cost so far: ' + str(np.sum(np.array(hist_cost))) + '\n')
            self.logger.write('    * time spent = ' + str(t1-t0) + '\n')
            self.logger.flush()
            
            
            res_trial = {}
            res_trial['hist_sim_opt'] = np.array(hist_simple_opt)
            res_trial['hist_sim_regrets'] = self.config['model']['SynData'].MfFn.maximum - np.array(hist_simple_opt)
            res_trial['hist_inf_opt'] = hist_infer_opt

            res_trial['hist_argm'] = np.array(hist_argm)
            res_trial['hist_argx'] = np.array(hist_argx)
            res_trial['hist_cum_cost'] = np.array(hist_cost)
            res_trial['hist_Fstar'] = np.array(hist_Fstar)


            res_mat_path = os.path.join(self.res_path, 'res')

            savemat(res_mat_path, res_trial, format='7.3')
            
        # end for
        
        tf.reset_default_graph()
        sess = tf.InteractiveSession()

        return hist_simple_opt, hist_infer_opt, hist_argm, hist_argx, hist_cost

    
    def eval_mf_query(self):
        
        opt = []
        Xopt = []
        
        for m in range(self.M):
            maximum = -np.inf
            maximiser = None
            for t in range(self.config['OptRandStart']):
                #
                self.sess.run(tf.compat.v1.variables_initializer(var_list=[self.Xquery]))
                #
                self.mutual_info_optimizers[m].minimize(self.sess)
                Xstar = self.sess.run(self.Xquery)
                info = -self.sess.run(self.neg_mutual_info[m], feed_dict={self.Xquery:Xstar})
                if info > maximum:
                    maximiser = Xstar
                    maximum = info
                # if
            # for
            opt.append(maximum)
            Xopt.append(maximiser)
        
        opt = np.array(opt)
        N_opt = opt / np.array(self.config['cost'])
        
        argm = np.argmax(N_opt)
        argX = Xopt[argm]
        
        return argm, argX, opt


        
        
        