import streamlit as st
import numpy as np
import time
import os
import json
import math
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import pulp
from concurrent.futures import ThreadPoolExecutor

STATE_FILE = 'ahrh_checkpoint.json'
NUM_WORKERS = max(1, (os.cpu_count() or 2)//2)

st.set_page_config(page_title="AHRH ILP/MILP/UFLP Solver", page_icon="🧠", layout="wide")

CONTACT_EMAIL = "zakibeny@gmail.com"
CONTACT_PHONE = "0021355614305 / 00213779679073"
CONTACT_FAX = "036495241"

def make_problem(obj, A, b, senses, var_types, lb=None, ub=None, original_type="ILP", meta=None):
    obj = np.asarray(obj, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = len(obj)
    m = len(b)
    lb = np.zeros(n, dtype=float) if lb is None else np.asarray(lb, dtype=float).reshape(-1)
    ub = np.array([1.0 if vt=='B' else np.inf for vt in var_types], dtype=float) if ub is None else np.asarray(ub, dtype=float).reshape(-1)
    return {"obj": obj, "A": A, "b": b, "senses": list(senses), "var_types": list(var_types), "lb": lb, "ub": ub, "original_type": original_type, "meta": meta or {}}

def _is_discrete(vt): return vt in ("B","I")

def _bound_or_none(v): return None if not np.isfinite(v) else float(v)

def _sanitize_value(v, lo, hi, vartype):
    if vartype=="B": return float(int(np.clip(np.round(v),0,1)))
    if vartype=="I": return float(int(np.clip(np.round(v), -1e18 if not np.isfinite(lo) else lo, 1e18 if not np.isfinite(hi) else hi)))
    return float(np.clip(v, -1e18 if not np.isfinite(lo) else lo, 1e18 if not np.isfinite(hi) else hi))

def _check_feasibility_numeric(x, problem, tol=1e-7):
    lhs = problem["A"] @ x
    for j, sense in enumerate(problem["senses"]):
        rhs = problem["b"][j]
        if sense=="<=" and lhs[j]>rhs+tol: return False
        if sense==">=" and lhs[j]<rhs-tol: return False
        if sense=="=" and abs(lhs[j]-rhs)>tol: return False
    return True

def evaluate_solution(x, problem, tol=1e-7):
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(problem["obj"])
    if len(x)!=n: raise ValueError("Candidate vector has wrong size")
    vt, lb, ub = problem["var_types"], problem["lb"], problem["ub"]
    fixed_vals, cont_idx = [None]*n, []
    for i in range(n):
        if _is_discrete(vt[i]): fixed_vals[i]=_sanitize_value(x[i], lb[i], ub[i], vt[i])
        else: cont_idx.append(i)
    if not cont_idx:
        x_full=np.array(fixed_vals,dtype=float)
        return (float(problem["obj"]@x_full), x_full) if _check_feasibility_numeric(x_full,problem) else (float('inf'),None)
    prob = pulp.LpProblem("MILP_EVAL", pulp.LpMinimize)
    vars_lp=[]
    for i in range(n):
        if _is_discrete(vt[i]): vars_lp.append(fixed_vals[i])
        else: vars_lp.append(pulp.LpVariable(f"x_{i}", lowBound=_bound_or_none(lb[i]), upBound=_bound_or_none(ub[i]), cat="Continuous"))
    prob += pulp.lpSum(problem["obj"][i]*vars_lp[i] for i in range(n))
    for j, sense in enumerate(problem["senses"]):
        expr=pulp.lpSum(problem["A"][j,i]*vars_lp[i] for i in range(n))
        if sense=="<=": prob += expr<=problem["b"][j]
        elif sense==">=": prob += expr>=problem["b"][j]
        elif sense=="=": prob += expr==problem["b"][j]
    solver=pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    if prob.status!=pulp.LpStatusOptimal: return float('inf'),None
    x_full=np.array([float(vars_lp[i]) if not _is_discrete(vt[i]) else vars_lp[i] for i in range(n)],dtype=float)
    return float(pulp.value(prob.objective)), x_full

def lp_relaxation(problem):
    n=len(problem["obj"])
    m=len(problem["b"])
    prob=pulp.LpProblem("LP_Relax", pulp.LpMinimize)
    x=[pulp.LpVariable(f"x_{i}", lowBound=_bound_or_none(problem["lb"][i]), upBound=_bound_or_none(problem["ub"][i]), cat="Continuous") for i in range(n)]
    prob+=pulp.lpSum(problem["obj"][i]*x[i] for i in range(n))
    for j in range(m): prob += pulp.lpSum(problem["A"][j,i]*x[i] for i in range(n)) <= problem["b"][j] if problem["senses"][j]=="<=" else (>=problem["b"][j] if problem["senses"][j]==">=" else ==problem["b"][j])
    solver=pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    if prob.status!=pulp.LpStatusOptimal: return None,None
    return np.array([float(pulp.value(x[i])) for i in range(n)],dtype=float), float(pulp.value(prob.objective))

def get_fractional_indices(x, problem, eps=1e-6):
    return np.array([i for i,vt in enumerate(problem["var_types"]) if vt in ("B","I") and abs(x[i]-np.round(x[i]))>eps],dtype=int)

def compute_R(x, problem):
    vals=[abs(x[i]-np.round(x[i])) for i,vt in enumerate(problem["var_types"]) if vt in ("B","I")]
    return max(vals) if vals else 0.0

def local_search(x, best_cost, problem):
    x = np.asarray(x, dtype=float).copy(); best_x=x.copy(); best=best_cost
    for i, vt in enumerate(problem["var_types"]):
        candidates=[1.0-round(best_x[i])] if vt=="B" else ([best_x[i]-1.0,best_x[i]+1.0] if vt=="I" else [] )
        for cand in candidates:
            x_new=best_x.copy(); x_new[i]=_sanitize_value(cand, problem["lb"][i], problem["ub"][i], vt)
            cost, x_feas = evaluate_solution(x_new, problem)
            if cost<best and x_feas is not None: best=cost; best_x=x_feas.copy()
    return best,best_x

def generate_biased_directions(x_lp, frac_idx, count, alpha, bias_strength=0.5):
    n_free=len(frac_idx); dirs=[]
    if x_lp is None or n_free==0: return np.random.randn(count,max(1,n_free))/np.linalg.norm(np.random.randn(count,max(1,n_free)),axis=1,keepdims=True)
    base_dir=x_lp[frac_idx]-0.5; base_dir=base_dir/np.linalg.norm(base_dir) if np.linalg.norm(base_dir)>0 else np.zeros(n_free)
    for _ in range(count): u=np.random.randn(n_free); u=u/np.linalg.norm(u); dirs.append((bias_strength*base_dir+(1-bias_strength)*u)/np.linalg.norm(bias_strength*base_dir+(1-bias_strength)*u))
    return np.array(dirs)

def hierarchical_radial_scan(x_center,R_val,frac_idx,problem,best_cost,best_x,x_lp=None):
    if len(frac_idx)==0: return best_cost,best_x
    x_frac=x_center[frac_idx].copy(); n_free=len(frac_idx); local_best=best_cost; local_best_x=best_x; alpha=R_val/(np.sqrt(n_free)+1e-12)
    dirs=generate_biased_directions(x_lp, frac_idx, 5, alpha)
    for u in dirs:
        for sign in [1,-1]:
            x_cand=x_frac+sign*alpha*u; x_cand_int=np.round(x_cand).astype(int); x_cand_int=np.maximum(x_cand_int,0)
            x_full=x_center.copy(); x_full[frac_idx]=x_cand_int
            cost,x_feas=evaluate_solution(x_full,problem)
            if cost<local_best and x_feas is not None: local_best=cost; local_best_x=x_feas.copy()
    return local_best,local_best_x

def vcycle(x,problem,coarse,x_lp,best_cost):
    frac_idx=get_fractional_indices(x,problem); R_val=compute_R(x,problem)
    new_cost,new_x=hierarchical_radial_scan(x,R_val,frac_idx,problem,best_cost,x,x_lp)
    if new_cost<best_cost: best_cost=new_cost; x=new_x
    if len(coarse)<=10 and len(coarse)>0:
        best_coarse=best_cost; best_x_coarse=x
        for bits in range(1<<len(coarse)):
            xc=np.array([(bits>>i)&1 for i in range(len(coarse))]); x_full=x.copy();
            for idx,val in zip(coarse,xc): x_full[idx]=val
            cost,x_feas=evaluate_solution(x_full,problem)
            if cost<best_coarse and x_feas is not None: best_coarse=cost; best_x_coarse=x_feas
        if best_coarse<best_cost: best_cost=best_coarse; x=best_x_coarse
    best_cost,x=local_search(x,best_cost,problem)
    return best_cost,x

def solve_ahrh(problem,max_cycles,k_coarse,patience,use_R,R_tol,stable_gap_needed,use_cost_repeat,cost_repeat_times,use_gap_repeat,gap_repeat_times,use_contraction,diff_tol,use_R_stability,R_stability_tol,R_stability_cycles,resume=False):
    n=len(problem["obj"]); x_lp,lp_val=lp_relaxation(problem)
    if x_lp is None: lp_val=float('inf')
    x=np.round(x_lp).astype(int) if x_lp is not None else np.zeros(n,dtype=int); x=np.clip(x,problem["lb"],problem["ub"])
    best_cost,x=evaluate_solution(x,problem); history=[]; total_time=0.0; start_cycle=1
    cycles_log=[]; gap_history=[]; R_history=[]; diff_history=[]; no_improve=0; cycles_done=0; stop_reason=""; acceleration_active=False; cost_repeat_count=0; gap_repeat_count=0; stable_gap_count=0; r_stable_count=0; last_cost=None; last_gap=None; last_R=None; last_x=x.copy(); start_time=time.time(); last_save_time=time.time()
    progress_bar=st.progress(0); status_placeholder=st.empty(); details_placeholder=st.empty(); time_placeholder=st.empty(); pause_button_placeholder=st.empty()
    for cycle in range(start_cycle,max_cycles+1):
        while st.session_state.get('paused',False): time.sleep(0.5)
        if x_lp is not None:
            open_now=np.where(x>0.5)[0].tolist(); top_lp=np.argsort(-x_lp)[:k_coarse].tolist(); coarse=list(set(open_now+top_lp))
            if len(coarse)>10: coarse=[i for i,_ in sorted([(i,x_lp[i]) for i in coarse], key=lambda x:x[1],reverse=True)[:10]]
        else: coarse=[]
        new_cost,new_x=vcycle(x,problem,coarse,x_lp,best_cost)
        if new_cost==float('inf') or new_x is None: new_cost=best_cost; new_x=x.copy()
        gap=(new_cost-lp_val)/lp_val*100 if lp_val not in [0,float('inf')] else 0; R_val=compute_R(new_x,problem); diff=np.linalg.norm(new_x-last_x); improved=new_cost<best_cost-1e-6
        if improved: best_cost=new_cost; x=new_x; no_improve=0
        else: no_improve+=1
        gap_history.append(gap); R_history.append(R_val); diff_history.append(diff)
        entry={'cycle':cycle,'cost':new_cost,'gap':gap,'R':R_val,'diff':diff,'improved':improved,'best_so_far':best_cost,'acceleration':acceleration_active}
        cycles_log.append(entry); progress=(cycle-start_cycle+1)/(max_cycles-start_cycle+1); progress_bar.progress(progress)
        elapsed=total_time+(time.time()-start_time)
        stop_now=False
        if no_improve>=patience: stop_reason=f"Patience ({patience})"; stop_now=True
        if not stop_now and use_R and R_val<R_tol:
            if last_gap is not None and abs(gap-last_gap)<1e-6: stable_gap_count+=1; stop_now=stable_gap_count>=stable_gap_needed; stop_reason=f"R<{R_tol} and gap stable" if stop_now else stop_reason
            else: stable_gap_count=0
        if not stop_now and use_cost_repeat and last_cost is not None and abs(new_cost-last_cost)<1e-6: cost_repeat_count+=1; stop_now=cost_repeat_count>=cost_repeat_times; stop_reason=f"Cost repeat" if stop_now else stop_reason
        if not stop_now and use_gap_repeat and last_gap is not None and abs(gap-last_gap)<1e-6: gap_repeat_count+=1; stop_now=gap_repeat_count>=gap_repeat_times; stop_reason=f"Gap repeat" if stop_now else stop_reason
        if not stop_now and use_contraction and diff<diff_tol and R_val<R_tol: stop_reason="Contraction"; stop_now=True
        if not stop_now and use_R_stability and len(R_history)>=2: prev_R=R_history[-2]; rel_change=abs(R_val-prev_R)/prev_R if prev_R>1e-12 else 0; r_stable_count+=1 if rel_change<R_stability_tol else 0; stop_now=r_stable_count>=R_stability_cycles; stop_reason=f"R stability" if stop_now else stop_reason
        last_cost=new_cost; last_gap=gap; last_R=R_val; last_x=new_x.copy()
        if stop_now: cycles_done=cycle; break
    if cycles_done==0: cycles_done=max_cycles; stop_reason=f"Max cycles ({max_cycles})"; total_time=elapsed
    progress_bar.empty(); status_placeholder.empty(); details_placeholder.empty(); time_placeholder.empty(); pause_button_placeholder.empty()
    return {'best_cost':best_cost,'lp_val':lp_val,'gap':(best_cost-lp_val)/lp_val*100 if lp_val not in [0,float('inf')] else 0,'cycles_done':cycles_done,'gap_history':gap_history,'R_history':R_history,'diff_history':diff_history,'cycles_log':cycles_log,'stop_reason':stop_reason,'acceleration_active':acceleration_active,'total_time':total_time,'open_fac':None}
