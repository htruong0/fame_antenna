# generally useful functions

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pyjet
from itertools import chain
from multiprocessing import Pool
from fame.utilities import njet_functions

class myException(Exception):
    pass


def convertToInt(my_list):
    '''Convert list to integers.'''
    return [int(item) for item in my_list]


def flatten(my_list):
    '''Flatten list of lists.'''
    return list(chain.from_iterable(my_list))


def complementary(num_jets, i, j):
    '''Find complementary element of list. Mainly used for finding the 'other' jets.'''
    nums = [k for k in range(1, num_jets+1)]
    return list(set(nums) - set([i, j]))


def dot(p1, p2):
    'Minkowski metric dot product of momenta in matrix form'
    if type(p1) != np.array or type(p2) != np.array:
        p1 = np.array(p1)
        p2 = np.array(p2)
    if len(p1.shape) == 1:
        return p1[0]*p2[0] - p1[1]*p2[1] - p1[2]*p2[2] - p1[3]*p2[3]
    elif len(p1.shape) == 2:
        return p1[:, 0]*p2[:, 0] - p1[:, 1]*p2[:, 1] - p1[:, 2]*p2[:, 2] - p1[:, 3]*p2[:, 3]
    elif len(p1.shape) == 3:
        return p1[:, :, 0]*p2[:, :, 0] - p1[:, :, 1]*p2[:, :, 1] - p1[:, :, 2]*p2[:, :, 2] - p1[:, :, 3]*p2[:, :, 3]

def check_kinematics(momenta, w):
    '''Check momenta is on-shell and momentum is conserved.'''
    num_jets = len(momenta[0]) - 2
    sum_mom = np.sum(momenta, axis=1)
    
    check_mom_cons = np.isclose(sum_mom, [2*w, 0, 0, 0], rtol=1E-5, atol=1E-5)
    if check_mom_cons.all() == True:
        print('\n######## Momentum conserved for all phase space points ##########\n')
    else:
        print('\n######## Not all phase space points conserve momentum  ##########')
        print('### {} / {} phase space points violate momentum conservation ###\n'.format(len(momenta - np.sum(check_mom_cons)), len(momenta)))
        
    max_idx = np.unravel_index(np.argmax(sum_mom, axis=None), sum_mom.shape)[0]
    
    print('Least momentum conserving phase space point = \n{}\n'.format(sum_mom[max_idx]-np.array([2*w, 0, 0, 0])))
    
    masses = dot(momenta, momenta)
    max_idx = np.unravel_index(np.argmax(masses, axis=None), masses.shape)[0]


    check_onshell = np.isclose(masses, np.zeros_like(masses), rtol=1E-5, atol=1E-5)
        
    if check_onshell.all() == True:
        print('##### All particles at each phase space point are on-shell ######\n')
    else:
        print('################# Not all particles are on-shell ################')
        print('### {} / {} phase space points violate on-shell condition ###\n'.format(len(momenta - np.sum(check_onshell)), len(momenta)))
        
    
    print('Most off-shell phase space point = \n{}'.format(masses[max_idx]))

    return "\n#################################################################"

def get_process_name(num_jets, mode='gluon'):
    '''Generate process name for number of jets.'''
    if mode == 'gluon':
        process_name = '_eeqq' + (num_jets-2)*'g'
    elif mode == 'quark':
        if num_jets < 4:
            raise myException("Less than 4 jets has to be in gluon mode.")
        process_name = '_eeddx' + 'uux'
        if num_jets > 4:
            process_name += (num_jets-4)*'g'
    else:
        raise NotImplementedError("Try mode=gluon or mode=quark.")
    return process_name


# clustering functions

def innerclustering(args):
    '''Multiprocessed clustering'''
    event, num_jets, w, y_cut, n = args
    
    idx = []
    algorithm = 'ee_kt'
    jet_def = pyjet.JetDefinition(algo=algorithm)
    sequence = pyjet.ClusterSequence(event, jet_def, ep=True)
    # dcut is the jet resolution, anything more collinear will be combined into a jet - defined to be 2*d_global_cut, or 0.01*s_com, whichever is larger.
    jet = sequence.exclusive_jets_dcut(dcut=y_cut*w**2)

    # selection conditions: only n-1 jets allowed and energy of jets has to be less than sqrt(s)/2
    conditions = [
        len(jet) >= num_jets - 1,
        all(event['E'] <= w / 2)
    ]

    # if all critera are met then keep point
    if all(conditions) == True:
        idx.append(n)
        
    return idx
    
    
def clustering(boosted_mom, w, num_jets, y_cut, num_cores):
    '''Clusters momenta with Fastjet returning only 'good' events.'''
    p = Pool(num_cores)
    
    # split momenta into chunks for parallel clustering
    events = boosted_mom[:, 2:, :].ravel().view(dtype=pyjet.DTYPE_EP)
    events = np.array_split(events, len(events) / num_jets)
    
    total_idx = []
    idx = p.map(
        innerclustering,
        [(event, num_jets, w, y_cut, n) for n, event in enumerate(events)]
    )
    total_idx.extend(flatten(idx))
        
    p.close()
    p.join()

    return total_idx

# plotting functions

def plot_diffs(y_true, y_preds, bins=np.linspace(-0.01, 0.01, 100), percentage=True, title=None):
    logbins = np.logspace(-5, 1, 50)
    
    diff = lambda y_true, y_pred: abs(y_true - y_pred) / y_true * 100
    ratio = lambda y_true, y_pred: np.log(y_pred / y_true)
    
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    ax[1].axvline(0, color='k', alpha=0.1)
    for name, pred in y_preds.items():
        if percentage:
            n = len(y_true)
            weights = np.ones(n) / n
            ax[0].hist(diff(y_true, pred), bins=logbins, weights=weights, histtype='step', label=name)
            ax[1].hist(ratio(y_true, pred), bins=bins, weights=weights, histtype='step', label=name)
        else:
            ax[0].hist(diff(y_true, pred), bins=logbins, histtype='step', label=name)
            ax[1].hist(ratio(y_true, pred), bins=bins, histtype='step', label=name)

    for axes in ax:
        if percentage:
            axes.yaxis.set_major_formatter(mtick.PercentFormatter(1))
            axes.set_ylabel("Percent of points")
        axes.legend(fontsize='small', loc='upper left')
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Absolute percentage difference")
    ax[1].set_xlabel(r"$\log(|\mathcal{M}|^{2}_{\mathrm{NN}}/|\mathcal{M}|^{2}_{\mathrm{NJet}})$")
    plt.tight_layout()
    if title is not None:
        plt.suptitle(title)
        plt.subplots_adjust(top=0.9)
    return fig, ax
