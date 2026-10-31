"""Reproducible benchmark: original AHRH versus feasibility fix and certified cuts.
The original MPS parser drops RHS, so every run uses the SAME correctly parsed MPS.
"""
import ast,dataclasses,hashlib,json,math,os,random,subprocess,sys,time,typing,types,urllib.request
from pathlib import Path
from multiprocessing import get_context
import numpy as np
from scipy.optimize import linprog

NEEDED=set('CanonicalProblem SearchConfig SolveResult SearchNode safe_float validate_problem prune_small_coefficients objective_value constraint_slacks is_feasible violation_measure clip_to_bounds load_problem load_mps_file solve_lp_relaxation round_mixed greedy_repair build_initial_solution random_direction lp_guided_directions layer_bounds alpha_max_for_bounds sample_candidate node_score radial_scan_node should_prune should_refine refine_node build_initial_nodes rank_nodes smart_ahrh_solve _ahrh_candidate_rank _ahrh_better ahrh_exact_branch_prune'.split())
def load_core(source):
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in NEEDED]
    name='ahrh_benchmark_core'
    if name not in sys.modules: sys.modules[name]=types.ModuleType(name)
    e=sys.modules[name].__dict__
    e.update({k:getattr(typing,k) for k in ('List','Optional','Dict','Tuple')})
    e.update({'__name__':name,'np':np,'linprog':linprog,'dataclass':dataclasses.dataclass,'field':dataclasses.field,'asdict':dataclasses.asdict,'math':math,'os':os,'random':random,'time':time,'json':json})
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    tree=ast.Module(body=[future]+nodes,type_ignores=[]);ast.fix_missing_locations(tree)
    exec(compile(tree,'ahrh-core.py','exec'),e)
    return dict(e)
def trial(q,label,source,problem,config):
    started=time.perf_counter()
    try:
        core=load_core(source)
        result=core['smart_ahrh_solve'](problem,core['SearchConfig'](**config))
        x=np.asarray(result.best_x,dtype=float)
        q.put({'label':label,'seconds':round(time.perf_counter()-started,3),
               'status':result.status,'violation':round(core['violation_measure'](problem,x),5),
               'feasible':bool(core['is_feasible'](problem,x)),'iterations':result.iterations})
    except Exception as ex:q.put({'label':label,'seconds':round(time.perf_counter()-started,3),'error':repr(ex)})
def timed(label,source,problem,config,limit=18):
    ctx=get_context('fork');q=ctx.Queue();p=ctx.Process(target=trial,args=(q,label,source,problem,config))
    started=time.perf_counter();p.start();p.join(limit)
    if p.is_alive():
        p.terminate();p.join(3)
        return {'label':label,'timeout_seconds':limit,'solved':False}
    if p.exitcode!=0:return {'label':label,'exit_code':p.exitcode,'seconds':round(time.perf_counter()-started,3)}
    return q.get(timeout=2)
def main():
    original=subprocess.check_output(['git','show','origin/main:app.py'],text=True)
    patched=Path('app.py').read_text()
    base=load_core(original);integrated=load_core(patched)
    url='https://miplib.zib.de/WebData/instances/supportcase30.mps.gz'
    target=Path('supportcase30.mps.gz')
    if not target.exists():urllib.request.urlretrieve(url,str(target))
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    assert digest=='a1f7a043032f386646820ef81680d69fe12407c8cf1fac856e217dc0f2a50e63', 'source MPS checksum mismatch'
    start=time.perf_counter();problem=integrated['load_problem'](str(target));parse_seconds=time.perf_counter()-start
    assert problem.n_vars()==1024 and problem.n_cons()==1032
    assert set(problem.b)=={-30.0,-24.0,30.0,24.0,-1.0},'RHS parsing broken'
    from ahrh_exact_pruners import _guard
    assert _guard(problem)[0] is not None,'exact cuts did not activate on actual MPS'
    caps={'random_seed':3,'layers':1,'initial_directions':1,'max_levels':1,
          'max_iterations':1,'max_active_nodes':1,'no_improvement_limit':1,
          'node_budget':1,'alpha_steps':3,'max_repair_steps':4}
    report={'mps_sha':digest,'corrected_mps_parse_seconds':round(parse_seconds,4),
            'parser_fixed_RHS':True,'identical_correct_problem_for_all_runs':True,'limits':caps,
            'runs':[
               timed('original_solver_corrected_RHS',original,problem,caps),
               timed('patched_without_cuts',patched,problem,{**caps,'exact_supportcase30_pruning':False}),
               timed('patched_with_cuts',patched,problem,{**caps,'exact_supportcase30_pruning':True})],
            'scope':'full solver call with tight caps, NOT uncapped solution time'}
    Path('ahrh_benchmark_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('AHRH_BENCH_RESULT_JSON='+json.dumps(report,ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
