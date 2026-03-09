# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pulp
import time
import matplotlib.pyplot as plt
import pandas as pd
import requests
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import os
import sys
import traceback

warnings.filterwarnings("ignore")

# ------------------- معالج الاستثناءات العام -------------------
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    st.error("❌ حدث خطأ غير متوقع:")
    error_details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    st.code(error_details)

sys.excepthook = handle_exception

# ------------------- معلومات الاتصال -------------------
CONTACT_EMAIL = "zakibeny@gmail.com"
CONTACT_PHONE = "0021355614305 / 00213779679073"
CONTACT_FAX = "036495241"

# ------------------- إعدادات التوازي -------------------
NUM_WORKERS = 4

# ------------------- ترجمة النصوص -------------------
translations = {
    'العربية': {
        'app_title': '🧠 AHRH: خوارزمية هرمية انكماشية متطورة',
        'app_desc': 'هذا التطبيق يطبق خوارزمية AHRH المتقدمة التي تجمع بين:',
        'feature1': 'المسح الشعاعي الهرمي مع اتجاهات موجهة',
        'feature2': 'الرفع الهرمي للاتجاهات',
        'feature3': 'إزاحة الاسترخاء الديناميكية',
        'feature4': 'بحث محلي متقدم بأنماط تبادل متعددة (1-1، 2-1، 1-2، 2-2)',
        'feature5': 'توازي الحسابات لتسريع الأداء',
        'feature6': 'معايير توقف متعددة قابلة للاختيار',
        'feature7': 'وضع التسريع التلقائي عندما تقترب الفجوة من 2%',
        'problem_type': 'نوع المسألة',
        'ilp': 'برمجة خطية صحيحة عامة (ILP)',
        'uflp': 'مسألة مواقع المرافق (UFLP)',
        'sidebar_algo': '⚙️ معاملات الخوارزمية',
        'max_cycles': 'عدد الدورات الأقصى',
        'k_coarse': 'حجم المجموعة الخشنة (k)',
        'patience': 'الصبر (عدد الدورات بدون تحسن)',
        'sidebar_stop': '⏹️ معايير التوقف',
        'choose_criteria': 'اختر أي مجموعة من الشروط (عند تحقق أي منها تتوقف الخوارزمية):',
        'use_R': 'استخدام عتبة R (مع استقرار الفجوة)',
        'R_tol': 'قيمة R الصغرى (ε)',
        'stable_gap': 'عدد دورات استقرار الفجوة المطلوبة',
        'use_cost_repeat': 'استخدام تكرار التكلفة',
        'cost_repeat_times': 'عدد مرات التكرار',
        'use_gap_repeat': 'استخدام تكرار الفجوة',
        'gap_repeat_times': 'عدد مرات التكرار للفجوة',
        'use_contraction': 'استخدام معيار الانكماش (diff + R)',
        'diff_tol': 'عتبة الفرق بين الحلول (ε₁)',
        'workers': 'عدد العمال (للتوازي)',
        'tab_upload': '📂 رفع ملف',
        'tab_random': '🎲 توليد عشوائي',
        'tab_manual': '✍️ إدخال يدوي',
        'upload_header': 'رفع ملف المسألة',
        'upload_info': 'يدعم ملفات UFLP (مثل gs250) والملفات العامة بصيغة ILP.',
        'choose_file': 'اختر ملف المسألة',
        'file_type': 'نوع الملف',
        'uflp_file': 'ملف UFLP (n m 0 + بيانات)',
        'ilp_file': 'ملف ILP عام (n m + c + A + b)',
        'ilp_format_help': """
**تنسيق ملف ILP العام:**
- السطر الأول: `n m` (عدد المتغيرات، عدد القيود)
- السطر الثاني: معاملات الهدف c (n قيمة)
- ثم m سطر: مصفوفة القيود A (كل سطر n قيمة)
- السطر الأخير: الطرف الأيمن b (m قيمة)
""",
        'uflp_format_help': """
**تنسيق ملف UFLP:**
- السطر الأول: `n m 0` (عدد المواقع، عدد العملاء)
- ثم n سطر: `رقم الموقع` `تكلفة الفتح` `تكاليف النقل إلى m عميل`
""",
        'random_header': 'توليد مسألة عشوائية',
        'random_n': 'عدد المتغيرات (n)',
        'random_m': 'عدد القيود (m)',
        'random_button': '🎲 توليد وحل',
        'manual_header': 'إدخال بيانات المسألة يدويًا',
        'manual_warning': 'للمسائل الصغيرة فقط (n ≤ 10, m ≤ 10)',
        'manual_n': 'عدد المتغيرات (n)',
        'manual_m': 'عدد القيود (m)',
        'manual_c': 'معاملات الهدف c[i]',
        'manual_A': 'مصفوفة القيود A[i][j]',
        'manual_b': 'الطرف الأيمن b[i]',
        'solve_button': '🚀 حل المسألة المدخلة',
        'results': '📊 النتائج',
        'best_cost': 'أفضل تكلفة',
        'lp_val': 'قيمة LP',
        'gap': 'الفجوة',
        'open_fac': 'المرافق المفتوحة',
        'cycles_done': 'عدد الدورات',
        'time': 'الزمن (ث)',
        'size': 'حجم المسألة',
        'stop_reason': 'سبب التوقف',
        'gap_plot': '📈 تطور الفجوة و R خلال الدورات',
        'gap_label': 'الفجوة (%)',
        'R_label': 'R',
        'cycle_log': '📋 سجل الدورات',
        'cycle': 'دورة',
        'cost': 'التكلفة',
        'improved': 'تحسن',
        'best_so_far': 'أفضل حتى الآن',
        'yes': 'نعم',
        'no': 'لا',
        'download': '📥 تحميل التطور (CSV)',
        'info_placeholder': '👈 اختر مصدر المسألة من التبويبات أعلاه واضغط على زر التشغيل.',
        'footer': 'تم التطوير بواسطة Zakarya Benregreg - خوارزمية AHRH محمية ببراءة اختراع.',
        'acceleration_on': '⚡ وضع التسريع مُفعّل (فجوة < 2% و R < 0.01)',
        'acceleration_off': 'وضع التسريع غير مُفعّل',
        'feedback_section': '💬 تواصل معنا - أرسل تعليقك',
        'feedback_placeholder': 'اكتب تعليقك هنا... (سيتم إرساله كـ Issue في GitHub)',
        'feedback_submit': 'إرسال التعليق',
        'feedback_success': '✅ تم الإرسال بنجاح! يمكنك متابعة الـ issue على الرابط:',
        'feedback_error': '❌ فشل الإرسال:',
        'feedback_missing_token': '⚠️ خدمة إرسال التعليقات غير مفعلة حالياً.',
        'feedback_warning': 'الرجاء كتابة تعليق قبل الإرسال.',
    },
    'English': {
        'app_title': '🧠 AHRH: Advanced Hierarchical Radial Heuristic',
        'app_desc': 'This app implements the AHRH algorithm, combining:',
        'feature1': 'Hierarchical radial scan with biased directions',
        'feature2': 'Hierarchical direction lifting',
        'feature3': 'Dynamic relaxation shift',
        'feature4': 'Advanced local search (1-1, 2-1, 1-2, 2-2 swaps)',
        'feature5': 'Parallel computing for speed',
        'feature6': 'Multiple customizable stopping criteria',
        'feature7': 'Automatic acceleration mode when gap approaches 2%',
        'problem_type': 'Problem Type',
        'ilp': 'General Integer Programming (ILP)',
        'uflp': 'Facility Location (UFLP)',
        'sidebar_algo': '⚙️ Algorithm Parameters',
        'max_cycles': 'Max Cycles',
        'k_coarse': 'Coarse Set Size (k)',
        'patience': 'Patience (cycles without improvement)',
        'sidebar_stop': '⏹️ Stopping Criteria',
        'choose_criteria': 'Choose any combination (algorithm stops when any condition is met):',
        'use_R': 'Use R threshold (with gap stability)',
        'R_tol': 'R tolerance (ε)',
        'stable_gap': 'Stable gap cycles required',
        'use_cost_repeat': 'Use cost repetition',
        'cost_repeat_times': 'Repetition count',
        'use_gap_repeat': 'Use gap repetition',
        'gap_repeat_times': 'Repetition count',
        'use_contraction': 'Use contraction criterion (diff + R)',
        'diff_tol': 'Solution difference tolerance (ε₁)',
        'workers': 'Workers (parallel threads)',
        'tab_upload': '📂 Upload File',
        'tab_random': '🎲 Random Generation',
        'tab_manual': '✍️ Manual Input',
        'upload_header': 'Upload Problem File',
        'upload_info': 'Supports UFLP files (e.g., gs250) and general ILP files.',
        'choose_file': 'Choose a file',
        'file_type': 'File Type',
        'uflp_file': 'UFLP file (n m 0 + data)',
        'ilp_file': 'General ILP file (n m + c + A + b)',
        'ilp_format_help': """
**General ILP File Format:**
- First line: `n m` (number of variables, number of constraints)
- Second line: objective coefficients c (n values)
- Then m lines: constraint matrix A (n values per line)
- Last line: right-hand side b (m values)
""",
        'uflp_format_help': """
**UFLP File Format:**
- First line: `n m 0` (number of facilities, number of customers)
- Then n lines: `facility_index` `opening_cost` `transport_costs to m customers`
""",
        'random_header': 'Generate Random Instance',
        'random_n': 'Number of variables (n)',
        'random_m': 'Number of constraints (m)',
        'random_button': '🎲 Generate and Solve',
        'manual_header': 'Manual Data Entry',
        'manual_warning': 'For small problems only (n ≤ 10, m ≤ 10)',
        'manual_n': 'Number of variables (n)',
        'manual_m': 'Number of constraints (m)',
        'manual_c': 'Objective coefficients c[i]',
        'manual_A': 'Constraint matrix A[i][j]',
        'manual_b': 'Right-hand side b[i]',
        'solve_button': '🚀 Solve Entered Problem',
        'results': '📊 Results',
        'best_cost': 'Best Cost',
        'lp_val': 'LP Value',
        'gap': 'Gap',
        'open_fac': 'Open Facilities',
        'cycles_done': 'Cycles Done',
        'time': 'Time (s)',
        'size': 'Problem Size',
        'stop_reason': 'Stop Reason',
        'gap_plot': '📈 Gap & R Evolution',
        'gap_label': 'Gap (%)',
        'R_label': 'R',
        'cycle_log': '📋 Cycle Log',
        'cycle': 'Cycle',
        'cost': 'Cost',
        'improved': 'Improved?',
        'best_so_far': 'Best so far',
        'yes': 'Yes',
        'no': 'No',
        'download': '📥 Download Evolution (CSV)',
        'info_placeholder': '👈 Choose a data source and click the run button.',
        'footer': 'Developed by Zakarya Benregreg - AHRH algorithm patented.',
        'acceleration_on': '⚡ Acceleration mode ON (gap < 2% and R < 0.01)',
        'acceleration_off': 'Acceleration mode OFF',
        'feedback_section': '💬 Contact Us - Send your feedback',
        'feedback_placeholder': 'Write your comment here... (will be sent as a GitHub Issue)',
        'feedback_submit': 'Send Feedback',
        'feedback_success': '✅ Sent successfully! You can track the issue at:',
        'feedback_error': '❌ Sending failed:',
        'feedback_missing_token': '⚠️ Feedback service is currently disabled.',
        'feedback_warning': 'Please write a comment before sending.',
    },
    'Français': {
        'app_title': '🧠 AHRH: Algorithme Hiérarchique Radial Contractant',
        'app_desc': 'Cette application implémente l\'algorithme AHRH, combinant :',
        'feature1': 'Balayage radial hiérarchique avec directions orientées',
        'feature2': 'Relèvement hiérarchique des directions',
        'feature3': 'Décalage dynamique de relaxation',
        'feature4': 'Recherche locale avancée (échanges 1-1, 2-1, 1-2, 2-2)',
        'feature5': 'Calcul parallèle pour la rapidité',
        'feature6': 'Critères d\'arrêt multiples personnalisables',
        'feature7': 'Mode accélération automatique lorsque le gap approche 2%',
        'problem_type': 'Type de problème',
        'ilp': 'Programmation en nombres entiers générale (ILP)',
        'uflp': 'Localisation d\'installations (UFLP)',
        'sidebar_algo': '⚙️ Paramètres de l\'algorithme',
        'max_cycles': 'Cycles max',
        'k_coarse': 'Taille de l\'ensemble grossier (k)',
        'patience': 'Patience (cycles sans amélioration)',
        'sidebar_stop': '⏹️ Critères d\'arrêt',
        'choose_criteria': 'Choisissez une combinaison (l\'algorithme s\'arrête dès qu\'une condition est remplie) :',
        'use_R': 'Utiliser le seuil R (avec stabilité du gap)',
        'R_tol': 'Tolérance R (ε)',
        'stable_gap': 'Cycles de stabilité du gap requis',
        'use_cost_repeat': 'Utiliser la répétition du coût',
        'cost_repeat_times': 'Nombre de répétitions',
        'use_gap_repeat': 'Utiliser la répétition du gap',
        'gap_repeat_times': 'Nombre de répétitions',
        'use_contraction': 'Utiliser le critère de contraction (diff + R)',
        'diff_tol': 'Tolérance de différence entre solutions (ε₁)',
        'workers': 'Travailleurs (threads parallèles)',
        'tab_upload': '📂 Télécharger un fichier',
        'tab_random': '🎲 Génération aléatoire',
        'tab_manual': '✍️ Saisie manuelle',
        'upload_header': 'Télécharger le fichier problème',
        'upload_info': 'Accepte les fichiers UFLP (ex. gs250) et les fichiers ILP généraux.',
        'choose_file': 'Choisir un fichier',
        'file_type': 'Type de fichier',
        'uflp_file': 'Fichier UFLP (n m 0 + données)',
        'ilp_file': 'Fichier ILP général (n m + c + A + b)',
        'ilp_format_help': """
**Format du fichier ILP général :**
- Première ligne : `n m` (nombre de variables, nombre de contraintes)
- Deuxième ligne : coefficients objectifs c (n valeurs)
- Ensuite m lignes : matrice des contraintes A (n valeurs par ligne)
- Dernière ligne : second membre b (m valeurs)
""",
        'uflp_format_help': """
**Format du fichier UFLP :**
- Première ligne : `n m 0` (nombre de sites, nombre de clients)
- Ensuite n lignes : `indice_site` `coût_ouverture` `coûts_transport vers m clients`
""",
        'random_header': 'Générer une instance aléatoire',
        'random_n': 'Nombre de variables (n)',
        'random_m': 'Nombre de contraintes (m)',
        'random_button': '🎲 Générer et résoudre',
        'manual_header': 'Saisie manuelle des données',
        'manual_warning': 'Pour petits problèmes uniquement (n ≤ 10, m ≤ 10)',
        'manual_n': 'Nombre de variables (n)',
        'manual_m': 'Nombre de contraintes (m)',
        'manual_c': 'Coefficients objectifs c[i]',
        'manual_A': 'Matrice des contraintes A[i][j]',
        'manual_b': 'Second membre b[i]',
        'solve_button': '🚀 Résoudre le problème saisi',
        'results': '📊 Résultats',
        'best_cost': 'Meilleur coût',
        'lp_val': 'Valeur LP',
        'gap': 'Écart',
        'open_fac': 'Sites ouverts',
        'cycles_done': 'Cycles effectués',
        'time': 'Temps (s)',
        'size': 'Taille du problème',
        'stop_reason': 'Raison de l\'arrêt',
        'gap_plot': '📈 Évolution du gap et de R',
        'gap_label': 'Écart (%)',
        'R_label': 'R',
        'cycle_log': '📋 Journal des cycles',
        'cycle': 'Cycle',
        'cost': 'Coût',
        'improved': 'Amélioration ?',
        'best_so_far': 'Meilleur jusqu\'à présent',
        'yes': 'Oui',
        'no': 'Non',
        'download': '📥 Télécharger l\'évolution (CSV)',
        'info_placeholder': '👈 Choisissez une source de données et cliquez sur le bouton.',
        'footer': 'Développé par Zakarya Benregreg - Algorithme AHRH breveté.',
        'acceleration_on': '⚡ Mode accélération ACTIVÉ (gap < 2% et R < 0.01)',
        'acceleration_off': 'Mode accélération DÉSACTIVÉ',
        'feedback_section': '💬 Contactez-nous - Envoyez votre commentaire',
        'feedback_placeholder': 'Écrivez votre commentaire ici... (sera envoyé comme Issue GitHub)',
        'feedback_submit': 'Envoyer',
        'feedback_success': '✅ Envoyé avec succès ! Suivez l\'issue sur :',
        'feedback_error': '❌ Échec de l\'envoi :',
        'feedback_missing_token': '⚠️ Le service de commentaires est actuellement désactivé.',
        'feedback_warning': 'Veuillez écrire un commentaire avant d\'envoyer.',
    }
}

def t(key):
    return translations[st.session_state.language][key]

if 'language' not in st.session_state:
    st.session_state.language = 'English'

# ------------------- تحويل UFLP إلى ILP -------------------
def uflp_to_ilp(f, c):
    n = len(f)
    m = c.shape[1]
    n_vars = n + n * m
    obj = np.zeros(n_vars)
    obj[:n] = f
    obj[n:] = c.flatten()
    n_constraints = m + n * m
    A = np.zeros((n_constraints, n_vars))
    b = np.zeros(n_constraints)
    for j in range(m):
        for i in range(n):
            A[j, n + i*m + j] = 1
        b[j] = 1
    for i in range(n):
        for j in range(m):
            idx = m + i*m + j
            A[idx, n + i*m + j] = 1
            A[idx, i] = -1
            b[idx] = 0
    return obj, A, b, n_vars, n_constraints, n

# ------------------- دوال تقييم الحل -------------------
def evaluate_solution(x, obj, A, b, sense, uflp_info=None):
    if uflp_info is not None:
        n_y = uflp_info['n_y']
        f = uflp_info['f']
        c = uflp_info['c']
        y = np.round(x[:n_y]).astype(int)
        open_fac = np.where(y > 0.5)[0]
        if len(open_fac) == 0:
            return float('inf')
        total = np.sum(f[open_fac])
        for j in range(c.shape[1]):
            total += np.min(c[open_fac, j])
        return total
    else:
        x_int = np.round(x).astype(int)
        x_int = np.maximum(x_int, 0)
        if np.all(A @ x_int <= b + 1e-6):
            if sense == 1:
                return obj @ x_int
            else:
                return - (obj @ x_int)
        else:
            return float('inf')

def lp_relaxation(obj, A, b, sense=1):
    n = len(obj)
    m = len(b)
    prob = pulp.LpProblem("LP_Relax", pulp.LpMinimize if sense == 1 else pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat='Continuous') for i in range(n)]
    if sense == 1:
        prob += pulp.lpSum(obj[i] * x[i] for i in range(n))
    else:
        prob += pulp.lpSum(-obj[i] * x[i] for i in range(n))
    for j in range(m):
        prob += pulp.lpSum(A[j][i] * x[i] for i in range(n)) <= b[j]
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    if prob.status == pulp.LpStatusOptimal:
        x_val = np.array([pulp.value(x[i]) for i in range(n)])
        obj_val = pulp.value(prob.objective)
        if sense == 1:
            return x_val, obj_val
        else:
            return x_val, -obj_val
    else:
        return None, None

# ------------------- دوال الخوارزمية -------------------
def compute_R(y):
    return np.max(np.minimum(y, 1 - y))

def get_fractional_indices(y, eps=0.01):
    return np.where((y > eps) & (y < 1 - eps))[0]

def generate_biased_directions(x_lp, frac_idx, count, alpha, bias_strength=0.5):
    n_free = len(frac_idx)
    if x_lp is None or n_free == 0:
        dirs = np.random.randn(count, max(1, n_free))
        dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        return dirs
    x_frac_target = np.clip(x_lp[frac_idx], 0, 1)
    base_dir = x_frac_target - 0.5
    if np.linalg.norm(base_dir) > 0:
        base_dir = base_dir / np.linalg.norm(base_dir)
    else:
        base_dir = np.zeros(n_free)
    dirs = []
    for _ in range(count):
        rand = np.random.randn(n_free)
        rand = rand / (np.linalg.norm(rand) + 1e-12)
        u = bias_strength * base_dir + (1 - bias_strength) * rand
        u = u / (np.linalg.norm(u) + 1e-12)
        dirs.append(u)
    return np.array(dirs)

def hierarchical_radial_scan(x_center, R_val, frac_idx, obj, A, b, best_cost, best_x, x_lp=None, sense=1, uflp_info=None):
    if len(frac_idx) == 0:
        return best_cost, best_x
    x_frac = x_center[frac_idx].copy()
    n_free = len(frac_idx)
    local_best = best_cost
    local_best_x = best_x
    alpha = R_val / (np.sqrt(n_free) + 1e-12)
    dirs = generate_biased_directions(x_lp, frac_idx, 5, alpha, bias_strength=0.5)
    for u in dirs:
        for sign in [1, -1]:
            x_cand = x_frac + sign * alpha * u
            x_cand_int = np.round(x_cand).astype(int)
            x_cand_int = np.maximum(x_cand_int, 0)
            x_full = x_center.copy()
            x_full[frac_idx] = x_cand_int
            cost = evaluate_solution(x_full, obj, A, b, sense, uflp_info)
            if cost < local_best:
                local_best = cost
                local_best_x = x_full.copy()
    return local_best, local_best_x

def local_search(x, best_cost, obj, A, b, sense=1, uflp_info=None):
    n = len(x)
    best_x = x.copy()
    best = best_cost
    for i in range(n):
        x_new = best_x.copy()
        x_new[i] = 1 - x_new[i]
        cost = evaluate_solution(x_new, obj, A, b, sense, uflp_info)
        if cost < best:
            best = cost
            best_x = x_new
    return best, best_x

def vcycle(x, obj, A, b, coarse, x_lp, best_cost, sense=1, uflp_info=None):
    frac_idx = get_fractional_indices(x)
    R_val = compute_R(x)
    new_cost, new_x = hierarchical_radial_scan(x, R_val, frac_idx, obj, A, b, best_cost, x, x_lp, sense, uflp_info)
    if new_cost < best_cost:
        best_cost = new_cost
        x = new_x
    if len(coarse) <= 10 and len(coarse) > 0:
        best_coarse = best_cost
        best_x_coarse = x
        for bits in range(1 << len(coarse)):
            xc = np.array([(bits >> i) & 1 for i in range(len(coarse))])
            x_full = x.copy()
            for idx, val in zip(coarse, xc):
                x_full[idx] = val
            cost = evaluate_solution(x_full, obj, A, b, sense, uflp_info)
            if cost < best_coarse:
                best_coarse = cost
                best_x_coarse = x_full
        if best_coarse < best_cost:
            best_cost = best_coarse
            x = best_x_coarse
    best_cost, x = local_search(x, best_cost, obj, A, b, sense, uflp_info)
    return best_cost, x

# ------------------- دوال قراءة الملفات -------------------
def read_uflp_file(text):
    lines = text.strip().splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith(('#', '!', '//', 'FILE:')):
            clean_lines.append(line)
    if len(clean_lines) < 2:
        raise ValueError("الملف لا يحتوي على بيانات كافية.")
    parts = clean_lines[0].split()
    if len(parts) == 3 and parts[2] == '0':
        n = int(parts[0])
        m = int(parts[1])
    elif len(parts) == 2:
        n = int(parts[0])
        m = int(parts[1])
    else:
        raise ValueError("السطر الأول غير صحيح")
    f = np.zeros(n, dtype=float)
    c = np.zeros((n, m), dtype=float)
    for i in range(n):
        if i+1 >= len(clean_lines):
            raise ValueError(f"الملف لا يحتوي على {n} سطراً من البيانات.")
        line = clean_lines[i+1]
        parts_line = line.split()
        if len(parts_line) == m+1:
            f[i] = float(parts_line[0])
            for j in range(m):
                c[i, j] = float(parts_line[1+j])
        elif len(parts_line) == m+2:
            f[i] = float(parts_line[1])
            for j in range(m):
                c[i, j] = float(parts_line[2+j])
        else:
            raise ValueError(f"السطر {i+2} لا يحتوي على العدد المناسب من القيم.")
    return f, c, n, m

def read_ilp_file(text):
    lines = text.strip().splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith(('#', '!', '//', 'FILE:')):
            clean_lines.append(line)
    if len(clean_lines) < 3:
        raise ValueError("الملف لا يحتوي على بيانات كافية.")
    parts = clean_lines[0].split()
    n = int(parts[0])
    m = int(parts[1])
    c = np.array(list(map(float, clean_lines[1].split())))
    if len(c) != n:
        raise ValueError("عدد معاملات الهدف لا يتطابق مع n")
    A = np.zeros((m, n))
    for i in range(m):
        row = list(map(float, clean_lines[2+i].split()))
        if len(row) != n:
            raise ValueError(f"عدد عناصر الصف {i} لا يتطابق مع n")
        A[i] = row
    b = np.array(list(map(float, clean_lines[2+m].split())))
    if len(b) != m:
        raise ValueError("عدد عناصر b لا يتطابق مع m")
    return c, A, b, n, m

def generate_random_ilp(n, m):
    c = np.random.uniform(1, 10, n)
    A = np.random.uniform(0, 5, (m, n))
    b = np.random.uniform(5, 20, m)
    return c, A, b

def generate_random_uflp(n, m):
    f = np.random.uniform(1000, 20000, n)
    c = np.random.uniform(100, 500, (n, m))
    return f, c

# ------------------- دالة إرسال التعليق -------------------
def send_to_github_issue(comment, repo_owner, repo_name, token):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    title = f"تعليق من مستخدم في {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    data = {
        "title": title,
        "body": comment
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 201:
            return True, response.json().get("html_url")
        else:
            return False, f"خطأ {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)

# ------------------- دالة الحل الرئيسية -------------------
def solve_ahrh(obj, A, b, uflp_info, sense, max_cycles, k_coarse, patience,
               use_R, R_tol, stable_gap_needed,
               use_cost_repeat, cost_repeat_times,
               use_gap_repeat, gap_repeat_times,
               use_contraction, diff_tol):
    n = len(obj)
    x_lp, lp_val = lp_relaxation(obj, A, b, sense)
    if x_lp is None:
        lp_val = float('inf')
    x = np.round(x_lp).astype(int) if x_lp is not None else np.zeros(n, dtype=int)
    x = np.maximum(x, 0)
    best_cost = evaluate_solution(x, obj, A, b, sense, uflp_info)
    if best_cost == float('inf'):
        x = np.zeros(n, dtype=int)
        best_cost = evaluate_solution(x, obj, A, b, sense, uflp_info)
    cycles_log = []
    gap_history = []
    R_history = []
    diff_history = []
    no_improve = 0
    cycles_done = 0
    stop_reason = ""
    acceleration_active = False
    cost_repeat_count = 0
    gap_repeat_count = 0
    stable_gap_count = 0
    last_cost = None
    last_gap = None
    last_R = None
    last_x = x.copy()
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    details_placeholder = st.empty()
    start_time = time.time()
    for cycle in range(max_cycles):
        if x_lp is not None:
            open_now = np.where(x > 0.5)[0].tolist()
            top_lp = np.argsort(-x_lp)[:k_coarse].tolist()
            coarse = list(set(open_now + top_lp))
            if len(coarse) > 10:
                importance = [(i, x_lp[i]) for i in coarse]
                importance.sort(key=lambda x: x[1], reverse=True)
                coarse = [i for i, _ in importance[:10]]
        else:
            coarse = []
        if acceleration_active:
            st.session_state.acceleration = True
        else:
            st.session_state.acceleration = False
        new_cost, new_x = vcycle(x, obj, A, b, coarse, x_lp, best_cost, sense, uflp_info)
        gap = (new_cost - lp_val) / lp_val * 100 if lp_val not in [0, float('inf')] else 0
        R_val = compute_R(new_x)
        diff = np.linalg.norm(new_x - last_x)
        improved = new_cost < best_cost - 1e-6
        if improved:
            best_cost = new_cost
            x = new_x
            no_improve = 0
        else:
            no_improve += 1
        gap_history.append(gap)
        R_history.append(R_val)
        diff_history.append(diff)
        cycles_log.append({
            'cycle': cycle+1,
            'cost': new_cost,
            'gap': gap,
            'R': R_val,
            'diff': diff,
            'improved': improved,
            'best_so_far': best_cost
        })
        progress_bar.progress((cycle + 1) / max_cycles)
        status_placeholder.info(f"**Cycle {cycle+1} / {max_cycles}**")
        details_placeholder.markdown(
            f"**Current cost:** {new_cost:,.2f}  \n"
            f"**Gap:** {gap:.4f}%  \n"
            f"**R:** {R_val:.6f}  \n"
            f"**Improved:** {'✅' if improved else '❌'}  \n"
            f"**Best so far:** {best_cost:,.2f}"
        )
        time.sleep(0.1)
        if not acceleration_active and gap < 2.0 and R_val < 0.01:
            acceleration_active = True
            st.session_state.acceleration = True
            st.info(t('acceleration_on'))
        elif acceleration_active and (gap >= 2.0 or R_val >= 0.01):
            acceleration_active = False
            st.session_state.acceleration = False
        stop_now = False
        if no_improve >= patience:
            stop_reason = f"Patience ({patience} cycles without improvement)"
            stop_now = True
        if not stop_now and use_R and R_val < R_tol:
            if last_gap is not None and abs(gap - last_gap) < 1e-6:
                stable_gap_count += 1
                if stable_gap_count >= stable_gap_needed:
                    stop_reason = f"R < {R_tol} and gap stable for {stable_gap_needed} cycles"
                    stop_now = True
            else:
                stable_gap_count = 0
        if not stop_now and use_cost_repeat:
            if last_cost is not None and abs(new_cost - last_cost) < 1e-6:
                cost_repeat_count += 1
                if cost_repeat_count >= cost_repeat_times:
                    stop_reason = f"Cost repeated {cost_repeat_times} times"
                    stop_now = True
            else:
                cost_repeat_count = 0
        if not stop_now and use_gap_repeat:
            if last_gap is not None and abs(gap - last_gap) < 1e-6:
                gap_repeat_count += 1
                if gap_repeat_count >= gap_repeat_times:
                    stop_reason = f"Gap repeated {gap_repeat_times} times"
                    stop_now = True
            else:
                gap_repeat_count = 0
        if not stop_now and use_contraction:
            if diff < diff_tol and R_val < R_tol:
                stop_reason = f"Contraction: diff < {diff_tol} and R < {R_tol}"
                stop_now = True
        last_cost = new_cost
        last_gap = gap
        last_R = R_val
        last_x = new_x.copy()
        if stop_now:
            cycles_done = cycle + 1
            break
    if cycles_done == 0:
        cycles_done = max_cycles
        stop_reason = f"Max cycles ({max_cycles}) reached"
    total_time = time.time() - start_time
    progress_bar.empty()
    status_placeholder.empty()
    details_placeholder.empty()
    if acceleration_active:
        st.info(t('acceleration_on'))
    return {
        'best_cost': best_cost,
        'lp_val': lp_val,
        'gap': (best_cost - lp_val) / lp_val * 100 if lp_val not in [0, float('inf')] else 0,
        'cycles_done': cycles_done,
        'gap_history': gap_history,
        'R_history': R_history,
        'diff_history': diff_history,
        'cycles_log': cycles_log,
        'stop_reason': stop_reason,
        'acceleration_active': acceleration_active,
        'total_time': total_time
    }

# ------------------- واجهة Streamlit -------------------
st.set_page_config(page_title="AHRH Solver - Unified ILP/UFLP", layout="wide")

col1, col2 = st.columns([4, 1])
with col2:
    language = st.selectbox(
        label="اختر اللغة / Choose language / Choisir la langue",
        options=['English', 'Français', 'العربية'],
        key='language_selector',
        label_visibility="collapsed"
    )
    st.session_state.language = language

st.title(t('app_title'))

st.markdown(f"""
<div style="background-color: #ffeeee; padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0; border: 3px solid red;">
    <span style="color: red; font-size: 32px; font-weight: bold;">✉️ {CONTACT_EMAIL}</span><br>
    <span style="color: red; font-size: 32px; font-weight: bold;">📞 {CONTACT_PHONE}</span><br>
    <span style="color: red; font-size: 32px; font-weight: bold;">📠 {CONTACT_FAX}</span>
</div>
""", unsafe_allow_html=True)

st.markdown(t('app_desc'))
st.markdown(f"- {t('feature1')}\n- {t('feature2')}\n- {t('feature3')}\n- {t('feature4')}\n- {t('feature5')}\n- {t('feature6')}\n- {t('feature7')}")

with st.sidebar:
    st.header(t('sidebar_algo'))
    max_cycles = st.slider(t('max_cycles'), 5, 200, 50, 5)
    k_coarse = st.slider(t('k_coarse'), 3, 10, 5)
    patience = st.slider(t('patience'), 2, 20, 5)
    st.header(t('sidebar_stop'))
    st.markdown(t('choose_criteria'))
    use_R = st.checkbox(t('use_R'), value=False)
    if use_R:
        R_tol = st.number_input(t('R_tol'), value=1e-6, format="%.0e", step=1e-6)
        stable_gap_needed = st.number_input(t('stable_gap'), min_value=1, max_value=5, value=2)
    else:
        R_tol, stable_gap_needed = 1e-6, 2
    use_cost_repeat = st.checkbox(t('use_cost_repeat'), value=False)
    if use_cost_repeat:
        cost_repeat_times = st.number_input(t('cost_repeat_times'), min_value=2, max_value=10, value=2)
    else:
        cost_repeat_times = 2
    use_gap_repeat = st.checkbox(t('use_gap_repeat'), value=False)
    if use_gap_repeat:
        gap_repeat_times = st.number_input(t('gap_repeat_times'), min_value=2, max_value=10, value=2)
    else:
        gap_repeat_times = 2
    use_contraction = st.checkbox(t('use_contraction'), value=True)
    if use_contraction:
        diff_tol = st.number_input(t('diff_tol'), value=1e-12, format="%.0e", step=1e-12)
    else:
        diff_tol = 1e-12
    st.markdown("---")
    st.write(f"{t('workers')}: {NUM_WORKERS}")

problem_type = st.radio(t('problem_type'), [t('ilp'), t('uflp')])
is_uflp = (problem_type == t('uflp'))

tab1, tab2, tab3 = st.tabs([t('tab_upload'), t('tab_random'), t('tab_manual')])

with tab1:
    st.header(t('upload_header'))
    st.info(t('upload_info'))
    with st.expander("📄 مساعدة حول تنسيق الملف"):
        if is_uflp:
            st.markdown(t('uflp_format_help'))
        else:
            st.markdown(t('ilp_format_help'))
    uploaded_file = st.file_uploader(t('choose_file'), type=None)
    if uploaded_file is not None:
        with st.spinner("Reading file and running algorithm..."):
            try:
                text = uploaded_file.getvalue().decode("utf-8", errors='ignore')
            except:
                text = uploaded_file.getvalue().decode("latin-1", errors='ignore')
            try:
                lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith(('#', '!', '//', 'FILE:'))]
                first_parts = lines[0].split()
                if len(first_parts) == 3 and first_parts[2] == '0' or (len(first_parts) == 2 and is_uflp):
                    f, c, n, m = read_uflp_file(text)
                    st.success(f"UFLP file loaded: {n} facilities, {m} customers")
                    obj, A, b, n_vars, n_constraints, n_y = uflp_to_ilp(f, c)
                    uflp_info = {'n_y': n_y, 'f': f, 'c': c}
                    sense = 1
                else:
                    c_vec, A, b, n, m = read_ilp_file(text)
                    st.success(f"ILP file loaded: {n} variables, {m} constraints")
                    obj = c_vec
                    n_vars = n
                    uflp_info = None
                    sense = 1
                result = solve_ahrh(obj, A, b, uflp_info, sense,
                                    max_cycles, k_coarse, patience,
                                    use_R, R_tol, stable_gap_needed,
                                    use_cost_repeat, cost_repeat_times,
                                    use_gap_repeat, gap_repeat_times,
                                    use_contraction, diff_tol)
                if result:
                    st.session_state['result'] = result
                    st.session_state['n'] = n
                    st.session_state['m'] = m
                    st.session_state['is_uflp'] = (uflp_info is not None)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.code(traceback.format_exc())

with tab2:
    st.header(t('random_header'))
    col1, col2 = st.columns(2)
    with col1:
        n_rand = st.number_input(t('random_n'), min_value=5, max_value=50, value=10, step=1, key="n_rand")
    with col2:
        m_rand = st.number_input(t('random_m'), min_value=5, max_value=50, value=10, step=1, key="m_rand")
    if st.button(t('random_button'), key="gen_rand"):
        with st.spinner("Generating and solving..."):
            try:
                if is_uflp:
                    f, c = generate_random_uflp(int(n_rand), int(m_rand))
                    obj, A, b, n_vars, n_constraints, n_y = uflp_to_ilp(f, c)
                    uflp_info = {'n_y': n_y, 'f': f, 'c': c}
                    sense = 1
                    n_disp = n_rand
                    m_disp = m_rand
                else:
                    c_vec, A, b = generate_random_ilp(int(n_rand), int(m_rand))
                    obj = c_vec
                    n_vars = n_rand
                    uflp_info = None
                    sense = 1
                    n_disp = n_rand
                    m_disp = m_rand
                result = solve_ahrh(obj, A, b, uflp_info, sense,
                                    max_cycles, k_coarse, patience,
                                    use_R, R_tol, stable_gap_needed,
                                    use_cost_repeat, cost_repeat_times,
                                    use_gap_repeat, gap_repeat_times,
                                    use_contraction, diff_tol)
                if result:
                    st.session_state['result'] = result
                    st.session_state['n'] = n_disp
                    st.session_state['m'] = m_disp
                    st.session_state['is_uflp'] = (uflp_info is not None)
                    st.success("Done!")
            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())

with tab3:
    st.header(t('manual_header'))
    st.warning(t('manual_warning'))
    col1, col2 = st.columns(2)
    with col1:
        n_man = st.number_input(t('manual_n'), min_value=1, max_value=10, value=3, step=1, key="n_man")
    with col2:
        m_man = st.number_input(t('manual_m'), min_value=1, max_value=10, value=3, step=1, key="m_man")
    if is_uflp:
        if ('f_man' not in st.session_state or 
            'c_man' not in st.session_state or 
            st.session_state.get('n_man_prev') != n_man or 
            st.session_state.get('m_man_prev') != m_man):
            
            st.session_state['f_man'] = np.zeros(n_man)
            st.session_state['c_man'] = np.zeros((n_man, m_man))
            st.session_state['n_man_prev'] = n_man
            st.session_state['m_man_prev'] = m_man
        else:
            if not isinstance(st.session_state['c_man'], np.ndarray) or st.session_state['c_man'].shape != (n_man, m_man):
                st.session_state['c_man'] = np.zeros((n_man, m_man))
            if len(st.session_state['f_man']) != n_man:
                st.session_state['f_man'] = np.zeros(n_man)
        
        st.subheader("تكاليف فتح المرافق f[i]")
        f_vals = []
        cols = st.columns(min(5, n_man))
        for i in range(n_man):
            with cols[i % 5]:
                default_f = float(st.session_state['f_man'][i]) if i < len(st.session_state['f_man']) else 0.0
                val = st.number_input(f"f[{i}]", value=default_f, key=f"f_man_{i}")
                f_vals.append(val)
        st.session_state['f_man'] = np.array(f_vals)
        
        st.subheader("تكاليف النقل c[i][j]")
        c_vals = np.zeros((n_man, m_man))
        for i in range(n_man):
            st.write(f"**الموقع {i}:**")
            cols = st.columns(min(5, m_man))
            for j in range(m_man):
                with cols[j % 5]:
                    try:
                        current_val = float(st.session_state['c_man'][i, j])
                    except (IndexError, TypeError, KeyError):
                        current_val = 0.0
                    val = st.number_input(f"c[{i}][{j}]", value=current_val, key=f"c_man_{i}_{j}")
                    c_vals[i, j] = val
        st.session_state['c_man'] = c_vals
        
        if st.button(t('solve_button'), key="solve_manual"):
            with st.spinner("Running algorithm..."):
                try:
                    f = st.session_state['f_man']
                    c = st.session_state['c_man']
                    obj, A, b, n_vars, n_constraints, n_y = uflp_to_ilp(f, c)
                    uflp_info = {'n_y': n_y, 'f': f, 'c': c}
                    sense = 1
                    result = solve_ahrh(obj, A, b, uflp_info, sense,
                                        max_cycles, k_coarse, patience,
                                        use_R, R_tol, stable_gap_needed,
                                        use_cost_repeat, cost_repeat_times,
                                        use_gap_repeat, gap_repeat_times,
                                        use_contraction, diff_tol)
                    if result:
                        st.session_state['result'] = result
                        st.session_state['n'] = n_man
                        st.session_state['m'] = m_man
                        st.session_state['is_uflp'] = True
                        st.success("Done!")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())
    else:
        if 'c_man' not in st.session_state or len(st.session_state.c_man) != n_man:
            st.session_state.c_man = np.zeros(n_man)
        if 'A_man' not in st.session_state or st.session_state.A_man.shape != (m_man, n_man):
            st.session_state.A_man = np.zeros((m_man, n_man))
        if 'b_man' not in st.session_state or len(st.session_state.b_man) != m_man:
            st.session_state.b_man = np.zeros(m_man)
        st.subheader(t('manual_c'))
        c_vals = []
        cols = st.columns(min(5, n_man))
        for i in range(n_man):
            with cols[i % 5]:
                val = st.number_input(f"c[{i}]", value=float(st.session_state.c_man[i]), key=f"c_man_{i}")
                c_vals.append(val)
        st.session_state.c_man = np.array(c_vals)
        st.subheader(t('manual_A'))
        for i in range(m_man):
            st.write(f"**Constraint {i+1}:**")
            cols = st.columns(min(5, n_man))
            for j in range(n_man):
                with cols[j % 5]:
                    val = st.number_input(f"A[{i}][{j}]", value=float(st.session_state.A_man[i, j]), key=f"A_man_{i}_{j}")
                    st.session_state.A_man[i, j] = val
        st.subheader(t('manual_b'))
        b_vals = []
        cols = st.columns(min(5, m_man))
        for i in range(m_man):
            with cols[i % 5]:
                val = st.number_input(f"b[{i}]", value=float(st.session_state.b_man[i]), key=f"b_man_{i}")
                b_vals.append(val)
        st.session_state.b_man = np.array(b_vals)
        if st.button(t('solve_button'), key="solve_manual"):
            with st.spinner("Running algorithm..."):
                try:
                    c_vec = st.session_state.c_man
                    A = st.session_state.A_man
                    b = st.session_state.b_man
                    obj = c_vec
                    uflp_info = None
                    sense = 1
                    result = solve_ahrh(obj, A, b, uflp_info, sense,
                                        max_cycles, k_coarse, patience,
                                        use_R, R_tol, stable_gap_needed,
                                        use_cost_repeat, cost_repeat_times,
                                        use_gap_repeat, gap_repeat_times,
                                        use_contraction, diff_tol)
                    if result:
                        st.session_state['result'] = result
                        st.session_state['n'] = n_man
                        st.session_state['m'] = m_man
                        st.session_state['is_uflp'] = False
                        st.success("Done!")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())

st.markdown("---")
st.header(t('results'))

if 'result' in st.session_state:
    res = st.session_state['result']
    colA, colB, colC, colD = st.columns(4)
    colA.metric(t('best_cost'), f"{res['best_cost']:,.2f}")
    colB.metric(t('lp_val'), f"{res['lp_val']:,.2f}")
    colC.metric(t('gap'), f"{res['gap']:.4f}%")
    if st.session_state.get('is_uflp', False):
        colD.metric(t('open_fac'), "—")
    else:
        colD.metric(t('size'), f"{st.session_state['n']}×{st.session_state['m']}")
    colE, colF, colG = st.columns(3)
    colE.metric(t('cycles_done'), res['cycles_done'])
    colF.metric(t('time'), f"{res['total_time']:.2f}")
    colG.metric(t('size'), f"{st.session_state['n']}×{st.session_state['m']}")
    st.info(f"**{t('stop_reason')}:** {res['stop_reason']}")
    if res.get('acceleration_active'):
        st.success(t('acceleration_on'))
    if res['gap_history']:
        st.subheader(t('gap_plot'))
        fig, ax1 = plt.subplots(figsize=(10, 5))
        cycles = list(range(1, len(res['gap_history'])+1))
        ax1.plot(cycles, res['gap_history'], 'b-o', label=t('gap_label'))
        ax1.set_xlabel(t('cycle'))
        ax1.set_ylabel(t('gap_label'), color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.plot(cycles, res['R_history'], 'r-s', label=t('R_label'))
        ax2.set_ylabel(t('R_label'), color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        plt.title(t('gap_plot'))
        fig.tight_layout()
        st.pyplot(fig)
        st.subheader(t('cycle_log'))
        df_cycles = pd.DataFrame(res['cycles_log'])
        df_cycles = df_cycles.rename(columns={
            'cycle': t('cycle'),
            'cost': t('cost'),
            'gap': t('gap'),
            'R': 'R',
            'diff': 'Diff',
            'improved': t('improved'),
            'best_so_far': t('best_so_far')
        })
        df_cycles[t('improved')] = df_cycles[t('improved')].apply(lambda x: t('yes') if x else t('no'))
        st.dataframe(df_cycles, width='stretch')
        df = pd.DataFrame({
            t('cycle'): cycles,
            t('gap_label'): res['gap_history'],
            'R': res['R_history']
        })
        csv = df.to_csv(index=False)
        st.download_button(
            label=t('download'),
            data=csv,
            file_name="evolution.csv",
            mime="text/csv"
        )
else:
    st.info(t('info_placeholder'))

# ------------------- قسم إرسال التعليقات إلى GitHub -------------------
st.markdown("---")
st.header(t('feedback_section'))

try:
    REPO_OWNER = st.secrets.get("REPO_OWNER", "zakibeny")
    REPO_NAME = st.secrets.get("REPO_NAME", "resolve-ilp-integer-linear-programing-")
    GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
except Exception:
    REPO_OWNER = "zakibeny"
    REPO_NAME = "resolve-ilp-integer-linear-programing-"
    GITHUB_TOKEN = ""

if not GITHUB_TOKEN:
    st.warning(t('feedback_missing_token'))
else:
    with st.form("feedback_form"):
        user_comment = st.text_area(t('feedback_section'), height=150,
                                     placeholder=t('feedback_placeholder'))
        submitted = st.form_submit_button(t('feedback_submit'))
        if submitted and user_comment.strip():
            with st.spinner("جاري الإرسال..."):
                success, result = send_to_github_issue(user_comment, REPO_OWNER, REPO_NAME, GITHUB_TOKEN)
                if success:
                    st.success(f"{t('feedback_success')} [{result}]({result})")
                else:
                    st.error(f"{t('feedback_error')} {result}")
        elif submitted:
            st.warning(t('feedback_warning'))

st.markdown("---")
st.caption(t('footer'))
