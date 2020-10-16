import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tqdm import tqdm

import ExpUtils as util


class MFNN:
    def __init__(self, config, TF_GPU_USAGE=0.25):
        self.config = config
        self.SynData = self.config['SynData']
        self.dim = self.SynData.dim
        self.M = self.SynData.Nfid
        self.MfData = self.SynData.data
        self.encode = self.config['feature']

        # Train/Test Input/Output holders
        self.tf_Xtrain_list = []
        self.tf_ytrain_list = []
        self.tf_Xtest_list = []
        self.tf_ytest_list = []
        for m in range(self.M):
            self.tf_Xtrain_list.append(tf.compat.v1.placeholder(util.tf_type, [None, self.dim]))
            self.tf_ytrain_list.append(tf.compat.v1.placeholder(util.tf_type, [None, 1]))
            self.tf_Xtest_list.append(tf.compat.v1.placeholder(util.tf_type, [None, self.dim]))
            self.tf_ytest_list.append(tf.compat.v1.placeholder(util.tf_type, [None, 1]))

        # Linear Mapping Weights
        self.tf_Wvar_Chol_list = []
        for m in range(self.M):
            Km = self.encode['Klist'][m]
            scale=1.0 # initialize with smaller values when there are numerical erros
            Lm = tf.linalg.band_part(scale*tf.Variable(tf.eye(Km+1), dtype=util.tf_type), -1, 0)

            self.tf_Wvar_Chol_list.append(Lm)

        # 
        self.tf_W_list = []
        self.tf_Wm_list = []
        for m in range(self.M):
            Km = self.encode['Klist'][m]
            dist_noise = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros([Km+1]), scale_diag=tf.ones([Km+1]))
            Wm = tf.Variable(tf.random.truncated_normal([Km+1,1]), dtype=util.tf_type)
            
            self.tf_Wm_list.append(Wm)
            self.tf_W_list.append(Wm+self.tf_Wvar_Chol_list[m]@tf.reshape(dist_noise.sample(),[-1,1]))

        # noise prior
        self.tf_log_gam_a = tf.Variable(-10, dtype=util.tf_type) 
        self.tf_log_gam_b = tf.Variable(-10, dtype=util.tf_type) 
        self.noise_gam_prior = tfp.distributions.Gamma(
            tf.exp(self.tf_log_gam_a), tf.exp(self.tf_log_gam_b)
        )
        
        # noise observations
        self.tf_tau_list = []
        for m in range(self.M):
            logtau_m = tf.Variable(0.0, dtype=util.tf_type)
            self.tf_tau_list.append(tf.exp(logtau_m))
            
        # initialize NN
        self.mf_encode_list = self.init_feature_encode(self.encode)   
        # concatenate NN with linear projection
        self.mf_outputs, self.mf_aug_features = self.init_mf_outputs(self.tf_Xtrain_list, self.tf_W_list)
        self.mf_pred_outputs, self.mf_pred_aug_features = self.init_mf_outputs(self.tf_Xtest_list, self.tf_W_list)


        self.expect_llh = self.eval_expect_llh()
        self.KL = self.eval_divergence()
        # negative evidence lower bound
        self.nelbo = -(self.expect_llh - self.KL)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.config['learning_rate'])
        self.minimizer = self.optimizer.minimize(self.nelbo)
        
        self.Xquery = tf.Variable(tf.random.uniform(minval=self.SynData.lb, maxval=self.SynData.ub, shape=[1,self.dim]), dtype=util.tf_type)
        
        self.tf_Ws_list = []
        for m in range(self.M):
            Km = self.encode['Klist'][m]
            self.tf_Ws_list.append(tf.compat.v1.placeholder(util.tf_type, [Km+1, 1]))
            
        self.ws_fstar, self.ws_aug_feature = self.mf_output(self.Xquery, self.M-1, self.tf_Ws_list)
        self.nfstar = -tf.squeeze(self.ws_fstar)
        
        self.nfstar_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.nfstar, 
                                                                method='L-BFGS-B', 
                                                                var_to_bounds={self.Xquery: [self.SynData.lb, self.SynData.ub]},
                                                                var_list=[self.Xquery],
                                                                options={'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'eps':self.config['Fstar']['rate'],
                                                                           'ftol' : 1.0 * np.finfo(float).eps},)
        
        # finding inference maximum
        self.Xinfer = tf.Variable(tf.random.uniform(minval=self.SynData.lb, maxval=self.SynData.ub, shape=[1,self.dim]), dtype=util.tf_type)
        self.infer_star, self.infer_aug_feature = self.mf_output(self.Xinfer, self.M-1, self.tf_W_list)
        self.neg_infer_maximum = -tf.squeeze(self.infer_star)
        self.neg_infer_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.neg_infer_maximum, 
                                                                method='L-BFGS-B', 
                                                                var_to_bounds={self.Xinfer: [self.SynData.lb, self.SynData.ub]},
                                                                var_list=[self.Xinfer],
                                                                options={'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'eps':self.config['Infer']['rate'],
                                                                           'ftol' : 1.0 * np.finfo(float).eps},)

        gpu_options =tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=TF_GPU_USAGE)

        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
                gpu_options=gpu_options,
            )
        )
        
        self.sess.run(tf.compat.v1.global_variables_initializer())
        
    def init_feature_encode(self, encode):
        """Initialize the feature encoding(NN) weights and biases"""
        feature_encode_list = []
        for m in range(self.M):
            if m == 0:
                layers = [self.dim] + encode['hlayers'][m] + [encode['Klist'][m]]
            else:
                layers = [self.dim+1] + encode['hlayers'][m] + [encode['Klist'][m]]
            # end if
            nn = util.EncodeNN(layers, init=encode['init'], activation=encode['activation'])
            feature_encode_list.append(nn)
        # end for
  
        return feature_encode_list

    def mf_output(self, X, m, Wlist):
        # base fidelity
        feature = self.mf_encode_list[0].forward(X)
        augment_feature = tf.pad(feature, tf.constant([[0,0],[0,1]]), constant_values=1.0)
        output = tf.matmul(augment_feature, Wlist[0])

        for l in range(1, m+1):
            augment_input = tf.concat([output, X], axis=1)
            feature = self.mf_encode_list[l].forward(augment_input)
            augment_feature = tf.pad(feature, tf.constant([[0,0],[0,1]]), constant_values=1.0)
            output = tf.matmul(augment_feature, Wlist[l])
        # end for

        return output, augment_feature
    
    def init_mf_outputs(self, Xlist, Wlist):
        
        outputs = []
        features = []
        for m in range(self.M):
            output, feature = self.mf_output(Xlist[m], m, Wlist)
            outputs.append(output)
            features.append(feature)
        
        return outputs, features
    
    def eval_divergence(self):
        expect = []
        for m in range(self.M):
            Km = self.encode['Klist'][m]
            Lm = self.tf_Wvar_Chol_list[m]
            mu = self.tf_W_list[m]
            
            log_det_Lm = -0.5*tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(Lm))))
            log_expect_m = -0.5*(Km+1)*tf.math.log(2*np.pi) -\
                0.5*(tf.linalg.trace(tf.matmul(Lm, tf.transpose(Lm))) + tf.reduce_sum(mu*mu))
            
            expect.append(log_det_Lm - log_expect_m)
            
        return tf.add_n(expect)

    
    def eval_expect_llh(self):
        expect = []
        Nlist = self.config['SynData'].Ntrain_list
        for m in range(self.M):
            Nm = Nlist[m]
            phi_m = self.mf_aug_features[m]
            mu_m = self.tf_W_list[m]
            Lm = self.tf_Wvar_Chol_list[m]

            tau_m = self.tf_tau_list[m]
            ym = self.tf_ytrain_list[m]
            
            LmLmT = tf.matmul(Lm, tf.transpose(Lm))
            mumuT = tf.matmul(mu_m, tf.transpose(mu_m))
            
            tr_phi_Lm_mu = tf.linalg.trace(tf.transpose(phi_m) @ phi_m @ (LmLmT + mumuT))
            ym_phi_mu = tf.squeeze(ym*ym - 2*ym*tf.matmul(phi_m, mu_m))
            
            expect_m = 0.5*Nm*tf.math.log(tau_m) - 0.5*tau_m*(tf.reduce_sum(ym_phi_mu) + tr_phi_Lm_mu) +\
                self.noise_gam_prior.log_prob(tau_m)
            expect.append(expect_m)
        # end for
        return tf.add_n(expect)
    
    
    def train(self):
        
        hist_train_err = []
        hist_test_err = []

        fdict = {}
        for m in range(self.M):
            Dm = self.MfData[m]
            fdict[self.tf_Xtrain_list[m]] = Dm['Xtrain']
            fdict[self.tf_ytrain_list[m]] = Dm['ytrain']
            fdict[self.tf_Xtest_list[m]] = Dm['Xtest']

        for it in tqdm(range(self.config['epochs'] + 1)):
            self.sess.run(self.minimizer, feed_dict = fdict)
            
            if it % 100 == 0:
                nelbo = self.sess.run(self.nelbo, feed_dict=fdict)
                mf_pred = self.sess.run(self.mf_pred_outputs, feed_dict=fdict)
                mf_pred_train = self.sess.run(self.mf_outputs, feed_dict=fdict)
                
                if self.config['verbose']:
                    print('it %d: nelbo = %.5f' % (it, nelbo))

                for m in range(self.M):
                    pred_m = mf_pred[m]
                    pred_m_train = mf_pred_train[m]
                    
                    ground_ytest = self.MfData[m]['ytest']
                    ground_ytrain = self.MfData[m]['ytrain']
                    
                    err_test = np.sqrt(np.mean(np.square(pred_m - ground_ytest)))
                    err_train = np.sqrt(np.mean(np.square(pred_m_train - ground_ytrain)))
                    
                    hist_train_err.append(err_train)
                    hist_test_err.append(err_test)
                    
                    if self.config['verbose'] or it == self.config['epochs']:
                        print('  - fid %d: train_nrmse = %.5f, test_nrmse = %.5f' % (m, err_train, err_test))
    
        
        Fstar, Xstar = self.collect_fstar()
        infer_opt, infer_optser = self.eval_infer_opt()

        return Fstar, infer_optser, self.sess

    def collect_fstar(self):
        Wpost = []
        Lpost = []
        for m in range(self.M):
            Wpost.append(self.sess.run(self.tf_W_list[m]))
            Lpost.append(self.sess.run(self.tf_Wvar_Chol_list[m]))

        Fstar = []
        Xstar = []

        for s in range(self.config['Fstar']['Ns']):
            fdict = {}
            for m in range(self.M):
                Ws = np.random.multivariate_normal(np.squeeze(Wpost[m]), np.matmul(Lpost[m], Lpost[m].T))
                fdict[self.tf_Ws_list[m]] = Ws.reshape([-1,1])
            
            Fs = []
            Xs = []
            for t in range(self.config['Fstar']['RandomStart']):
                self.sess.run(tf.compat.v1.variables_initializer(var_list=[self.Xquery]))
                self.nfstar_optimizer.minimize(self.sess, feed_dict=fdict)
                fstar = -self.sess.run(self.nfstar, feed_dict=fdict)
                xstar = self.sess.run(self.Xquery, feed_dict=fdict)
                
                Fs.append(fstar)
                Xs.append(xstar)
                
            argx = np.argmax(np.array(Fs))
            
            Fstar.append(Fs[argx])
            Xstar.append(Xs[argx])

        return Fstar, Xstar
    
    
    def eval_infer_opt(self):
            
        infer_opt = []
        infer_optser = []
        for t in range(self.config['Infer']['RandomStart']):
            self.sess.run(tf.compat.v1.variables_initializer(var_list=[self.Xinfer]))
            self.neg_infer_optimizer.minimize(self.sess)
            infer_fstar = -self.sess.run(self.neg_infer_maximum)
            infer_xstar = self.sess.run(self.Xinfer)

            infer_opt.append(infer_fstar)
            infer_optser.append(infer_xstar)
    
        argx = np.argmax(np.array(infer_opt))
        
        return np.mean(np.array(infer_opt)), infer_optser[argx]


    
    
