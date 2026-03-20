from __future__ import annotations
import numpy as np, pulp, time, os
import streamlit as st
from datetime import datetime

STATE_FILE = "ahrh_checkpoint.json"

def make_problem(obj, A, b, senses, var_types, lb=None, ub=None):
    obj = np.asarray(obj, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = len(obj)
    m = len(b)
    lb = np.zeros(n, dtype=float) if lb is None else np.asarray(lb, dtype=float).reshape(-1)
    ub = np.ones(n, dtype=float) if ub is None else np.asarray(ub, dtype=float).reshape(-1)
    return {"obj": obj, "A": A, "b": b, "senses": list(senses), "var_types": list(var_types), "lb": lb, "ub": ub}

def uflp_to_ilp(f, c):
    f = np.asarray(f).reshape(-1)
    c = np.asarray(c)
    n = len(f)
    m = c.shape[1]
    n_vars = n + n*m
    obj = np.concatenate([f, c.reshape(-1)])
    A, b, senses = [], [], []
    for j in range(m):
        row = np.zeros(n_vars)
        for i in range(n):
            row[n + i*m + j] = 1
        A.append(row); b.append(1); senses.append("=")
    for i in range(n):
        for j in range(m):
            row = np.zeros(n_vars); row[n + i*m + j] = 1; row[i] = -1
            A.append(row); b.append(0); senses.append("<=")
    var_types = ["B"] * n_vars
    return make_problem(obj, np.array(A), np.array(b), senses, var_types)

def evaluate_solution(x, problem, tol=1e-7):
    x = np.asarray(x, dtype=float)
    n = len(problem["obj"])
    vt = problem["var_types"]
    lb, ub, obj = problem["lb"], problem["ub"], problem["obj"]
    fixed_vals = x.copy()
    for i in range(n):
        if vt[i] in ("B","I"):
            fixed_vals[i] = float(int(np.clip(round(fixed_vals[i]), lb[i], ub[i])))
    lhs = problem["A"] @ fixed_vals
    for j, sense in enumerate(problem["senses"]):
        if sense == "<=" and lhs[j] > problem["b"][j]+tol: return float('inf'), None
        if sense == ">=" and lhs[j] < problem["b"][j]-tol: return float('inf'), None
        if sense == "=" and abs(lhs[j]-problem["b"][j]) > tol: return float('inf'), None
    return float(obj @ fixed_vals), fixed_vals

def lp_relaxation(problem):
    n, m = len(problem["obj"]), len(problem["b"])
    prob = pulp.LpProblem("LP_Relax", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", lowBound=problem["lb"][i], upBound=problem["ub"][i], cat="Continuous") for i in range(n)]
    prob += pulp.lpSum(problem["obj"][i]*x[i] for i in range(n))
    for j in range(m):
        expr = pulp.lpSum(problem["A"][j,i]*x[i] for i in range(n))
        if problem["senses"][j]=="<=": prob += expr <= problem["b"][j]
        elif problem["senses"][j]==">=": prob += expr >= problem["b"][j]
        elif problem["senses"][j]=="=": prob += expr == problem["b"][j]
    pulp.PULP_CBC_CMD(msg=False).solve(prob)
    x_val = np.array([float(pulp.value(x[i])) for i in range(n)])
    obj_val = float(pulp.value(prob.objective))
    return x_val, obj_val

def get_fractional_indices(x, problem, eps=1e-6):
    idx=[]; x=np.asarray(x)
    for i, vt in enumerate(problem["var_types"]):
        if vt in ("B","I") and abs(x[i]-round(x[i]))>eps: idx.append(i)
    return np.array(idx,dtype=int)

def compute_R(x, problem):
    x = np.asarray(x)
    vals = [abs(x[i]-round(x[i])) for i,vt in enumerate(problem["var_types"]) if vt in ("B","I")]
    return max(vals) if vals else 0.0

def local_search(x, best_cost, problem):
    x=x.copy(); best_x=x.copy(); best=best_cost
    for i, vt in enumerate(problem["var_types"]):
        candidates=[]
        if vt=="B": candidates=[1.0-round(best_x[i])]
        elif vt=="I": candidates=[best_x[i]-1,best_x[i]+1]
        for cand in candidates:
            x_new=best_x.copy(); x_new[i]=np.clip(round(cand),problem["lb"][i],problem["ub"][i])
            cost, xf = evaluate_solution(x_new, problem)
            if cost<best and xf is not None: best=cost; best_x=xf.copy()
    return best, best_x

def hierarchical_radial_scan(x_center,R_val,frac_idx,problem,best_cost,best_x,x_lp=None):
    if len(frac_idx)==0: return best_cost,best_x
    x_frac=x_center[frac_idx].copy(); n_free=len(frac_idx); local_best=best_cost; local_best_x=best_x
    alpha = R_val/(np.sqrt(n_free)+1e-12)
    dirs = np.random.randn(5,n_free)
    dirs = dirs/np.linalg.norm(dirs,axis=1,keepdims=True)
    for u in dirs:
        for sign in [1,-1]:
            x_cand = x_frac + sign*alpha*u
            x_cand_int=np.round(np.clip(x_cand,0,1)).astype(int)
            x_full=x_center.copy(); x_full[frac_idx]=x_cand_int
            cost, xf=evaluate_solution(x_full, problem)
            if cost<local_best and xf is not None: local_best=cost; local_best_x=xf.copy()
    return local_best, local_best_x

def vcycle(x, problem, coarse, x_lp, best_cost):
    frac_idx=get_fractional_indices(x, problem); R_val=compute_R(x,problem)
    best_cost,new_x=hierarchical_radial_scan(x,R_val,frac_idx,problem,best_cost,x,x_lp)
    best_cost,new_x=local_search(new_x,best_cost,problem)
    return best_cost,new_x

def solve_ahrh(problem,max_cycles=100):
    n=len(problem["obj"]); x_lp, lp_val=lp_relaxation(problem)
    if x_lp is None: lp_val=float('inf')
    x = np.round(x_lp).astype(int) if x_lp is not None else np.zeros(n)
    best_cost,x = evaluate_solution(x,problem)
    history=[]; gap_history=[]; R_history=[]
    for cycle in range(1,max_cycles+1):
        coarse=[]
        best_cost,x = vcycle(x,problem,coarse,x_lp,best_cost)
        gap = (best_cost-lp_val)/lp_val*100 if lp_val not in [0,float('inf')] else 0
        R_val=compute_R(x,problem)
        gap_history.append(gap); R_history.append(R_val)
    return {"best_cost":best_cost,"lp_val":lp_val,"gap":gap,"gap_history":gap_history,"R_history":R_history}

st.set_page_config(page_title="AHRH Solver", layout="wide")
st.title("AHRH Solver ILP/MILP/UFLP")
uploaded_file = st.file_uploader("Upload problem file")
if uploaded_file is not None:
    text=uploaded_file.getvalue().decode("utf-8",errors="ignore")
    try:
        # تجربة gs250a/UFLP/ILP
        problem=None
        lines=text.strip().splitlines(); nums=[]
        for line in lines:
            if line.strip() and not line.startswith(('#','!','//','%','*')):
                nums.extend([float(x) for x in line.strip().split()])
        if len(nums)>=3:
            n=int(nums[0]); m=int(nums[1]); c=nums[2:2+n]; idx=2+n
            A=np.zeros((m,n)); b=np.zeros(m)
            for i in range(m):
                A[i]=nums[idx:idx+n]; idx+=n
            b=nums[idx:idx+m]
            problem=make_problem(np.array(c),A,np.array(b),['<=']*m,['I']*n)
        if problem is not None:
            result=solve_ahrh(problem,50)
            st.write("Best Cost:",result['best_cost'])
            st.line_chart({"Gap":result['gap_history'],"R":result['R_history']})
    except Exception as e:
        st.error(str(e))