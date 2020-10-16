import numpy as np
from hdf5storage import loadmat
from hdf5storage import savemat
import time
import argparse

import os
import sys
sys.path.append('../')

from BayesOpt import BayesOptMFMM
from SynFuncs import SynMfData
import ExpUtils as util



import tensorflow as tf
tf.get_logger().setLevel('WARNING')

# OMP error on MacOS
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

def dump_config(fname, config):
    
    dumper = open(fname, 'w+')  
    dumper.write('* domain: ' + config['model']['domain'] + '\n')
    dumper.write('* seed: ' + str(config['model']['SynData'].seed) + '\n')
    dumper.write('* Ntrain_init: ' + str(config['model']['SynData'].Ntrain_list) + '\n')
    dumper.write('* epochs: ' + str(config['model']['epochs']) + '\n')
    dumper.write('* activation: ' + config['model']['feature']['activation'] + '\n')
    dumper.write('* hlayers: ' + str(config['model']['feature']['hlayers']) + '\n')
    dumper.write('* Klayers: ' + str(config['model']['feature']['Klist']) + '\n')
    dumper.write('* num Fstar: ' + str(config['model']['Fstar']['Ns']) + '\n')
    dumper.write('* costs: ' + str(config['cost']) + '\n')
    dumper.write('* random start: ' + str(config['OptRandStart']) + '\n')
    dumper.write('* max opt iters: ' + str(config['maxiter']) + '\n')
    dumper.close()

def run(args):

    domain = args.domain
    Ntrain = [int(s) for s in args.inits.split(',')]
    if domain == 'Branin' or domain == 'Hartmann3D' or domain == 'Levy':
        Nfid = 3
        costs = [1,10,100]
    else:
        Nfid = 2
        costs = [1,10]
    
    Ntest = [1]*Nfid
    T = int(args.T)
    epochs = int(args.epochs)
    maxIter = int(args.maxIter)
    
    hw = [int(s) for s in args.hw.split(',')]
    hl = [int(s) for s in args.hl.split(',')]
    kl = [int(s) for s in args.kl.split(',')]
    
    hlayers = []
    Klist = []
    for m in range(Nfid):
        hlayers.append([hw[m]]*hl[m])
        Klist.append(kl[m])

    for t in range(T):     
        
        seed = np.random.randint(0,10000)

        SynData = SynMfData(domain, Ntrain.copy(), Ntest.copy(), seed=seed,perturb_scale=1e-2, perturb_thresh=1e-3)

        res_path = os.path.join('results', domain, 'MFBOMM', 'trial' + str(t))
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        logfname = os.path.join(res_path,'log-' + domain + '.txt')
        logger = open(logfname, 'w+')  

        config = {
            'logger': logger,
            'model': {
                'domain': domain,
                'SynData': SynData,
                'learning_rate' : 1e-3,
                'epochs' : epochs,
                'verbose' : False,
                'feature' : {
                    'model': 'NN',
                    'activation' : args.activation,
                    'init' : 'xavier',
                    'hlayers' : hlayers,
                    'Klist' : Klist,
                },
                'Fstar':{
                    'Ns':10,
                    'rate':1e-4,
                    'RandomStart':10,
                },
                'Infer': {
                    'rate':1e-4,
                    'RandomStart':10,
                },
            },
            'NQuad': 5,
            'OptInfoStep': 1e-4,
            'OptRegretStep': 1e-4, 
            'OptRandStart': 10,
            'cost': costs,
            'maxiter': maxIter,
        }
        
        config_fname = os.path.join(res_path,'config-' + domain + '.txt')
        dump_config(config_fname, config)

        logger.write('=====================' + 'Starting experiment ' + domain + ' trail#' + str(t+1) + '=====================\n')
        logger.flush()

        ##########################
        t0 = time.time()

        BO = BayesOptMFMM(config, res_path, TF_GPU_USAGE=0.2)
        hist_simplet_opt, hist_infer_opt, hist_argm, hist_argx, hist_cost = BO.optimize()

        t1 = time.time()
        #########################
        logger.write(' Finished trial, time spent = ' + str(t1-t0) + '\n')
        logger.flush()


        logger.close()
    
if __name__== "__main__" :
    
    args = argparse.ArgumentParser()
    args.add_argument("--domain", "-d", dest="domain", type=str, required=True)
    args.add_argument("--inits", "-i", dest="inits", type=str, required=True)
    args.add_argument("--epochs", "-e", dest="epochs", type=str, required=True)
    args.add_argument("--maxiter", "-t", dest="maxIter", type=str, required=True)
    args.add_argument("--activation", "-a", dest="activation", type=str, required=True)
    args.add_argument("--hlayers_width", '-w', dest='hw', type=str, required=True)
    args.add_argument("--hlayers_depth", '-l', dest='hl', type=str, required=True)
    args.add_argument("--klayers", '-k', dest='kl', type=str, required=True)
    args.add_argument("--trials", "-r", dest="T", type=str, required=True)
    
    args = args.parse_args()

    run(args)
