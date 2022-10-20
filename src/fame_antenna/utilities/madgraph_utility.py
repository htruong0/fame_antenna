import sys

def load_madgraph():
    mg_path = "/mt/home/htruong/software/MG5_aMC_v3_2_0/processes/emep_ddxng_loop/"
    sys.path.append(mg_path + "SubProcesses/")
    import allmatrix2py
    allmatrix2py.initialise(mg_path + "/Cards/param_card.dat")
    allmatrix2py.set_madloop_path(mg_path + "SubProcesses/MadLoop5_resources")
    return allmatrix2py


def parse_pdgs(pdgs_order, pids, num_jets):
    idx = num_jets - 3
    return pdgs_order[idx][:num_jets+2], pids[idx]

def get_me(p, alphas, mu2):
    return allmatrix2py.smatrixhel(
        pdg,
        pid,
        p.T,
        alphas,
        mu2,
        -1
    )

def parallel_get_me_pbar(momenta, alphas, mu2, num_cores):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm.auto import tqdm
    N = len(momenta)
    num_jets = momenta.shape[1]-2
    global allmatrix2py, pdg, pid
    allmatrix2py = load_madgraph()
    pdgs, pids = allmatrix2py.get_pdg_order()
    pdg, pid = parse_pdgs(pdgs, pids, num_jets)
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        with tqdm(total = N) as pbar:
            futures = {}
            for i, p in enumerate(momenta):
                future = executor.submit(get_me, p, alphas[i], mu2[i])
                futures[future] = i
            results = [None]*N
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                pbar.update(1)
    # don't need MadGraph return code
    res = [sublist[0] for sublist in results]
    return res