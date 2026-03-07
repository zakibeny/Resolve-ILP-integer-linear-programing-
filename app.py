# -*- coding: utf-8 -*-
"""
تطبيق AHRH لحل مسائل UFLP فقط
مع واجهة متعددة اللغات، عرض التقدم، معايير توقف متعددة، وإرسال التعليقات إلى GitHub Issues
(نسخة مبسطة ومخصصة لـ UFLP - تم إزالة LP العام)
"""

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
warnings.filterwarnings("ignore")

# ------------------- معلومات الاتصال -------------------
CONTACT_EMAIL = "zakibeny@gmail.com"
CONTACT_PHONE = "0021355614305 / 00213779679073"
CONTACT_FAX = "036495241"

# ------------------- إعدادات التوازي -------------------
NUM_WORKERS = 4

# ------------------- ترجمة النصوص (عربي/إنجليزي/فرنسي) -------------------
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
        'upload_info': 'يدعم ملفات UFLP بصيغة OR-Library (txt, dat, bub, opt, ...). يتم تجاهل الأسطر التي تبدأ بـ # أو ! أو FILE:',
        'choose_file': 'اختر ملف المسألة',
        'random_header': 'توليد مسألة UFLP عشوائية',
        'random_n': 'عدد المواقع (n)',
        'random_m': 'عدد العملاء (m)',
        'random_button': '🎲 توليد وحل',
        'manual_header': 'إدخال بيانات المسألة يدويًا',
        'manual_warning': 'للمسائل الصغيرة فقط (n ≤ 10, m ≤ 10)',
        'manual_n': 'عدد المواقع (n)',
        'manual_m': 'عدد العملاء (m)',
        'manual_f': 'تكاليف فتح المرافق f[i]',
        'manual_c': 'تكاليف النقل c[i][j]',
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
        'upload_header': 'Upload UFLP Problem File',
        'upload_info': 'Accepts UFLP files in OR-Library format (txt, dat, bub, opt, ...). Lines starting with #, !, or FILE: are ignored.',
        'choose_file': 'Choose a file',
        'random_header': 'Generate Random UFLP Instance',
        'random_n': 'Number of facilities (n)',
        'random_m': 'Number of customers (m)',
        'random_button': '🎲 Generate and Solve',
        'manual_header': 'Manual Data Entry',
        'manual_warning': 'For small problems only (n ≤ 10, m ≤ 10)',
        'manual_n': 'Number of facilities (n)',
        'manual_m': 'Number of customers (m)',
        'manual_f': 'Facility opening costs f[i]',
        'manual_c': 'Transportation costs c[i][j]',
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
        'upload_header': 'Télécharger un fichier UFLP',
        'upload_info': 'Accepte les fichiers UFLP au format OR-Library (txt, dat, bub, opt, ...). Les lignes commençant par #, ! ou FILE: sont ignorées.',
        'choose_file': 'Choisir un fichier',
        'random_header': 'Générer une instance UFLP aléatoire',
        'random_n': 'Nombre de sites (n)',
        'random_m': 'Nombre de clients (m)',
        'random_button': '🎲 Générer et résoudre',
        'manual_header': 'Saisie manuelle des données',
        'manual_warning': 'Pour petits problèmes uniquement (n ≤ 10, m ≤ 10)',
        'manual_n': 'Nombre de sites (n)',
        'manual_m': 'Nombre de clients (m)',
        'manual_f': 'Coûts d\'ouverture f[i]',
        'manual_c': 'Coûts de transport c[i][j]',
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

# ------------------- دوال UFLP (النسخة القديمة) -------------------
def solve_lp_fixed_y_uflp(y_int, f, c):
    open_fac = np.where(y_int > 0.5)[0]
    if len(open_fac) == 0:
        return float('inf')
    total = np.sum(f[open_fac])
    for j in range(c.shape[1]):
        total += min(c[i, j] for i in open_fac)
    return total

def lp_relaxation_uflp(f, c):
    n, m = len(f), c.shape[1]
    lp = pulp.LpProblem("UFLP_LP_Relax", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y_{i}", 0, 1) for i in range(n)]
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", 0) for i in range(n) for j in range(m)}
    lp += pulp.lpSum(f[i] * y[i] for i in range(n)) + pulp.lpSum(c[i, j] * x[(i, j)] for i in range(n) for j in range(m))
    for j in range(m):
        lp += pulp.lpSum(x[(i, j)] for i in range(n)) == 1
    for i in range(n):
        for j in range(m):
            lp += x[(i, j)] <= y[i]
    solver = pulp.PULP_CBC_CMD(msg=False)
    lp.solve(solver)
    if lp.status == 1:
        y_val = np.array([pulp.value(y[i]) for i in range(n)])
        obj = pulp.value(lp.objective)
        return y_val, obj
    return None, None

def get_fractional_indices(y, eps=0.01):
    return np.where((y > eps) & (y < 1 - eps))[0]

def compute_R(y):
    return np.max(np.minimum(np.abs(y), np.abs(1 - y)))

def generate_biased_directions(y_lp, frac_idx, count, alpha, bias_strength=0.5):
    n_free = len(frac_idx)
    if y_lp is None or n_free == 0:
        dirs = np.random.randn(count, max(1, n_free))
        dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        return dirs
    y_frac_target = np.clip(y_lp[frac_idx], 0, 1)
    base_dir = y_frac_target - 0.5
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

def hierarchical_radial_scan_parallel(y_center, R, n_free, f, c, best_cost, best_y,
                                      n_layers=2, dirs_per_layer=10, alpha_schedule='adaptive',
                                      y_lp=None, gap_threshold=5.0):
    frac_idx = get_fractional_indices(y_center)
    if len(frac_idx) == 0:
        return best_cost, best_y
    y_frac = y_center[frac_idx].copy()
    n_free_actual = len(frac_idx)
    local_best = best_cost
    local_best_y = best_y
    base_alpha = R / (np.sqrt(n_free) + 1e-12) if n_free > 0 else 0.1

    for layer in range(1, n_layers + 1):
        if alpha_schedule == 'adaptive':
            alpha_k = base_alpha * np.exp(- (layer - 1) / n_layers)
        else:
            alpha_k = (layer / n_layers) * base_alpha
        if hasattr(st.session_state, 'acceleration') and st.session_state.acceleration:
            dirs_count = max(2, dirs_per_layer // 2)
        else:
            dirs_count = max(3, dirs_per_layer // 2)
        dirs = generate_biased_directions(y_lp, frac_idx, dirs_count, alpha_k, bias_strength=0.5)

        def evaluate_direction(u):
            best_loc = local_best
            best_loc_y = local_best_y
            for sign in [1, -1]:
                y_cand = y_frac + sign * alpha_k * u
                y_cand_int = (y_cand > 0.5).astype(int)
                y_full = y_center.copy()
                y_full[frac_idx] = y_cand_int
                y_int = np.round(y_full).astype(int)
                cost = solve_lp_fixed_y_uflp(y_int, f, c)
                if cost < best_loc:
                    best_loc = cost
                    best_loc_y = y_int.copy()
            return best_loc, best_loc_y

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(evaluate_direction, dirs[i]) for i in range(dirs_count)]
            for future in as_completed(futures):
                cost, y_cand = future.result()
                if cost < local_best:
                    local_best = cost
                    local_best_y = y_cand
        y_center = local_best_y.copy()
        y_frac = y_center[frac_idx].copy()
    return local_best, local_best_y

def smooth(y, f, c, y_lp=None, iters=1, gap_threshold=5.0):
    best = solve_lp_fixed_y_uflp(y, f, c)
    best_y = y.copy()
    for _ in range(iters):
        R_val = compute_R(y)
        n_free = len(get_fractional_indices(y))
        new_cost, new_y = hierarchical_radial_scan_parallel(
            y, R_val, n_free, f, c, best, best_y,
            n_layers=2, dirs_per_layer=8, alpha_schedule='adaptive',
            y_lp=y_lp, gap_threshold=gap_threshold
        )
        if new_cost < best:
            best = new_cost
            best_y = new_y
        else:
            break
    return best, best_y

def local_search_advanced(y, best_cost, f, c, max_iter=10):
    n = len(f)
    improved = True
    iteration = 0
    best_y = y.copy()
    best = best_cost

    while improved and iteration < max_iter:
        improved = False
        open_fac = np.where(best_y > 0.5)[0].tolist()
        closed_fac = np.where(best_y < 0.5)[0].tolist()

        # 1-1 exchange
        for i in open_fac:
            for j in closed_fac:
                y_new = best_y.copy()
                y_new[i] = 0
                y_new[j] = 1
                cost = solve_lp_fixed_y_uflp(y_new, f, c)
                if cost < best:
                    best = cost
                    best_y = y_new
                    improved = True
                    open_fac = np.where(best_y > 0.5)[0].tolist()
                    closed_fac = np.where(best_y < 0.5)[0].tolist()
                    break
            if improved:
                break

        if not improved and len(open_fac) >= 2 and len(closed_fac) >= 1:
            # 2-1 exchange
            for i1 in range(len(open_fac)):
                for i2 in range(i1+1, len(open_fac)):
                    for j in closed_fac:
                        y_new = best_y.copy()
                        y_new[open_fac[i1]] = 0
                        y_new[open_fac[i2]] = 0
                        y_new[j] = 1
                        cost = solve_lp_fixed_y_uflp(y_new, f, c)
                        if cost < best:
                            best = cost
                            best_y = y_new
                            improved = True
                            break
                    if improved: break
                if improved: break

        if not improved and len(open_fac) >= 1 and len(closed_fac) >= 2:
            # 1-2 exchange
            for i in open_fac:
                for j1 in range(len(closed_fac)):
                    for j2 in range(j1+1, len(closed_fac)):
                        y_new = best_y.copy()
                        y_new[i] = 0
                        y_new[closed_fac[j1]] = 1
                        y_new[closed_fac[j2]] = 1
                        cost = solve_lp_fixed_y_uflp(y_new, f, c)
                        if cost < best:
                            best = cost
                            best_y = y_new
                            improved = True
                            break
                    if improved: break
                if improved: break

        if not improved and len(open_fac) >= 2 and len(closed_fac) >= 2:
            # 2-2 exchange
            for i1 in range(len(open_fac)):
                for i2 in range(i1+1, len(open_fac)):
                    for j1 in range(len(closed_fac)):
                        for j2 in range(j1+1, len(closed_fac)):
                            y_new = best_y.copy()
                            y_new[open_fac[i1]] = 0
                            y_new[open_fac[i2]] = 0
                            y_new[closed_fac[j1]] = 1
                            y_new[closed_fac[j2]] = 1
                            cost = solve_lp_fixed_y_uflp(y_new, f, c)
                            if cost < best:
                                best = cost
                                best_y = y_new
                                improved = True
                                break
                        if improved: break
                    if improved: break
                if improved: break

        iteration += 1

    return best, best_y

def vcycle(y, f, c, coarse, y_lp=None, gap_threshold=5.0):
    cost1, y1 = smooth(y, f, c, y_lp=y_lp, iters=1, gap_threshold=gap_threshold)
    if not coarse:
        return cost1, y1
    best = cost1
    best_y = y1
    n_coarse = len(coarse)

    def evaluate_bits(bits):
        yc = np.array([(bits >> i) & 1 for i in range(n_coarse)])
        y_full = y1.copy()
        for idx, val in zip(coarse, yc):
            y_full[idx] = val
        cost = solve_lp_fixed_y_uflp(y_full, f, c)
        return cost, y_full

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(evaluate_bits, bits) for bits in range(1 << n_coarse)]
        for future in as_completed(futures):
            cost, y_cand = future.result()
            if cost < best:
                best = cost
                best_y = y_cand
    cost2, y2 = smooth(best_y, f, c, y_lp=y_lp, iters=1, gap_threshold=gap_threshold)
    cost3, y3 = local_search_advanced(y2, cost2, f, c, max_iter=5)
    return cost3, y3

def solve_ahrh_uflp(f, c, max_cycles, k_coarse, patience,
                    use_R, R_tol, stable_gap_needed,
                    use_cost_repeat, cost_repeat_times,
                    use_gap_repeat, gap_repeat_times,
                    use_contraction, diff_tol):
    n, m = len(f), c.shape[1]
    y_lp, lp_val = lp_relaxation_uflp(f, c)
    if lp_val is None:
        lp_val = float('inf')

    y = np.ones(n, dtype=int)
    best = solve_lp_fixed_y_uflp(y, f, c)

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
    last_y = y.copy()

    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    details_placeholder = st.empty()
    start_time = time.time()

    for cycle in range(max_cycles):
        if y_lp is not None:
            open_now = np.where(y > 0.5)[0].tolist()
            top_lp = np.argsort(-y_lp)[:k_coarse].tolist()
            coarse = list(set(open_now + top_lp))
            if len(coarse) > 10:
                importance = [(i, y_lp[i]) for i in coarse]
                importance.sort(key=lambda x: x[1], reverse=True)
                coarse = [i for i, _ in importance[:10]]
        else:
            coarse = []

        if acceleration_active:
            st.session_state.acceleration = True
        else:
            st.session_state.acceleration = False

        new_cost, new_y = vcycle(y, f, c, coarse, y_lp=y_lp, gap_threshold=3.0)
        gap = (new_cost - lp_val) / lp_val * 100 if lp_val != float('inf') else 0
        R_val = compute_R(new_y)
        diff = np.linalg.norm(new_y - last_y)

        improved = new_cost < best - 1e-6
        if improved:
            best = new_cost
            y = new_y
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
            'best_so_far': best
        })

        progress_bar.progress((cycle + 1) / max_cycles)
        status_placeholder.info(f"**Cycle {cycle+1} / {max_cycles}**")
        details_placeholder.markdown(
            f"**Current cost:** {new_cost:,.2f}  \n"
            f"**Gap:** {gap:.4f}%  \n"
            f"**R:** {R_val:.6f}  \n"
            f"**Improved:** {'✅' if improved else '❌'}  \n"
            f"**Best so far:** {best:,.2f}"
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
        last_y = new_y.copy()

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
        'best_cost': best,
        'lp_val': lp_val,
        'gap': (best - lp_val) / lp_val * 100 if lp_val != float('inf') else 0,
        'open_fac': len(np.where(y > 0.5)[0]),
        'cycles_done': cycles_done,
        'gap_history': gap_history,
        'R_history': R_history,
        'diff_history': diff_history,
        'cycles_log': cycles_log,
        'stop_reason': stop_reason,
        'acceleration_active': acceleration_active,
        'total_time': total_time
    }

# ------------------- دوال قراءة ملفات UFLP -------------------
def read_uflp_file(text):
    lines = text.strip().splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('!') and not line.upper().startswith('FILE:'):
            clean_lines.append(line)
    if len(clean_lines) < 2:
        raise ValueError("الملف لا يحتوي على بيانات كافية.")
    n, m = None, None
    data_start = -1
    for idx, line in enumerate(clean_lines):
        parts = line.split()
        if len(parts) >= 2:
            try:
                n_cand = int(parts[0])
                m_cand = int(parts[1])
                if n_cand > 0 and m_cand > 0:
                    n, m = n_cand, m_cand
                    data_start = idx
                    break
            except:
                continue
    if n is None or m is None:
        raise ValueError("لم يتم العثور على سطر صالح يحتوي على n و m.")
    f = np.zeros(n, dtype=float)
    c = np.zeros((n, m), dtype=float)
    for i in range(n):
        if data_start + 1 + i >= len(clean_lines):
            raise ValueError(f"الملف لا يحتوي على {n} سطراً من البيانات.")
        line = clean_lines[data_start + 1 + i]
        parts = line.split()
        if len(parts) == 1 + m:
            f[i] = float(parts[0])
            for j in range(m):
                c[i, j] = float(parts[1 + j])
        elif len(parts) >= 2 + m:
            idx = int(parts[0]) - 1
            f[idx] = float(parts[1])
            for j in range(m):
                c[idx, j] = float(parts[2 + j])
        else:
            raise ValueError(f"السطر {data_start+1+i+1} لا يحتوي على العدد المناسب من القيم.")
    return f, c, n, m

def generate_random_uflp(n, m):
    f = np.random.uniform(1000, 20000, n)
    c = np.random.uniform(100, 500, (n, m))
    return f, c

# ------------------- دالة إرسال التعليق إلى GitHub Issues -------------------
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

# ------------------- واجهة Streamlit -------------------
st.set_page_config(page_title="AHRH Solver - UFLP", layout="wide")

col1, col2 = st.columns([4, 1])
with col2:
    language = st.selectbox("", ['English', 'Français', 'العربية'], key='language_selector')
    st.session_state.language = language

st.title(t('app_title'))

# معلومات الاتصال
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
    max_cycles = st.slider(t('max_cycles'), 5, 50, 15, 5)
    k_coarse = st.slider(t('k_coarse'), 3, 10, 5)
    patience = st.slider(t('patience'), 2, 10, 3)

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

tab1, tab2, tab3 = st.tabs([t('tab_upload'), t('tab_random'), t('tab_manual')])

with tab1:
    st.header(t('upload_header'))
    st.info(t('upload_info'))
    
    # شرح تنسيق الملف (اختياري)
    with st.expander("📄 مساعدة حول تنسيق الملف"):
        st.markdown("""
        **تنسيق ملف UFLP (مثل gs250, capb):**
        - السطر الأول: `n m 0` حيث n عدد المواقع، m عدد العملاء.
        - ثم n سطر، كل سطر يمثل موقعاً: `[رقم الموقع] [تكلفة الفتح] [تكلفة النقل إلى العميل 1] [تكلفة النقل إلى العميل 2] ... [تكلفة النقل إلى العميل m]`
        - مثال:
        ```
        5 3 0
        1 100 5 6 7
        2 150 8 9 10
        3 120 4 5 6
        4 130 7 8 9
        5 110 3 4 5
        ```
        - يتم تجاهل الأسطر التي تبدأ بـ # أو ! أو FILE:.
        """)
    
    uploaded_file = st.file_uploader(t('choose_file'), type=None)
    if uploaded_file is not None:
        with st.spinner("Reading file and running algorithm..."):
            try:
                text = uploaded_file.getvalue().decode("utf-8")
            except:
                text = uploaded_file.getvalue().decode("latin-1")
            try:
                f, c, n, m = read_uflp_file(text)
                st.success(f"File loaded: {n} facilities, {m} customers")
                result = solve_ahrh_uflp(
                    f, c,
                    max_cycles, k_coarse, patience,
                    use_R, R_tol, stable_gap_needed,
                    use_cost_repeat, cost_repeat_times,
                    use_gap_repeat, gap_repeat_times,
                    use_contraction, diff_tol
                )
                if result:
                    st.session_state['result'] = result
                    st.session_state['n'] = n
                    st.session_state['m'] = m
            except Exception as e:
                st.error(f"Error reading file: {e}")

with tab2:
    st.header(t('random_header'))
    col1, col2 = st.columns(2)
    with col1:
        n_rand = st.number_input(t('random_n'), min_value=5, max_value=200, value=50, step=5, key="n_rand")
    with col2:
        m_rand = st.number_input(t('random_m'), min_value=5, max_value=200, value=50, step=5, key="m_rand")
    if st.button(t('random_button'), key="gen_rand"):
        with st.spinner("Generating and solving..."):
            f, c = generate_random_uflp(int(n_rand), int(m_rand))
            result = solve_ahrh_uflp(
                f, c,
                max_cycles, k_coarse, patience,
                use_R, R_tol, stable_gap_needed,
                use_cost_repeat, cost_repeat_times,
                use_gap_repeat, gap_repeat_times,
                use_contraction, diff_tol
            )
            if result:
                st.session_state['result'] = result
                st.session_state['n'] = n_rand
                st.session_state['m'] = m_rand
                st.success("Done!")

with tab3:
    st.header(t('manual_header'))
    st.warning(t('manual_warning'))
    col1, col2 = st.columns(2)
    with col1:
        n_man = st.number_input(t('manual_n'), min_value=1, max_value=10, value=3, step=1, key="n_man")
    with col2:
        m_man = st.number_input(t('manual_m'), min_value=1, max_value=10, value=3, step=1, key="m_man")

    if 'f_man' not in st.session_state or st.session_state.get('n_man_prev') != n_man:
        st.session_state['f_man'] = np.zeros(n_man)
        st.session_state['n_man_prev'] = n_man
    if 'c_man' not in st.session_state or st.session_state.get('n_man_prev') != n_man or st.session_state.get('m_man_prev') != m_man:
        st.session_state['c_man'] = np.zeros((n_man, m_man))
        st.session_state['m_man_prev'] = m_man

    st.subheader(t('manual_f'))
    f_vals = []
    cols = st.columns(min(5, n_man))
    for i in range(n_man):
        with cols[i % 5]:
            val = st.number_input(f"f[{i}]", value=float(st.session_state['f_man'][i]), key=f"f_man_{i}")
            f_vals.append(val)
    st.session_state['f_man'] = np.array(f_vals)

    st.subheader(t('manual_c'))
    c_vals = np.zeros((n_man, m_man))
    for i in range(n_man):
        st.write(f"**{t('manual_c')} {i}:**")
        cols = st.columns(min(5, m_man))
        for j in range(m_man):
            with cols[j % 5]:
                val = st.number_input(f"c[{i}][{j}]", value=float(st.session_state['c_man'][i, j]), key=f"c_man_{i}_{j}")
                c_vals[i, j] = val
    st.session_state['c_man'] = c_vals

    if st.button(t('solve_button'), key="solve_manual"):
        with st.spinner("Running algorithm..."):
            result = solve_ahrh_uflp(
                st.session_state['f_man'], st.session_state['c_man'],
                max_cycles, k_coarse, patience,
                use_R, R_tol, stable_gap_needed,
                use_cost_repeat, cost_repeat_times,
                use_gap_repeat, gap_repeat_times,
                use_contraction, diff_tol
            )
            if result:
                st.session_state['result'] = result
                st.session_state['n'] = n_man
                st.session_state['m'] = m_man
                st.success("Done!")

st.markdown("---")
st.header(t('results'))

if 'result' in st.session_state:
    res = st.session_state['result']
    colA, colB, colC, colD = st.columns(4)
    colA.metric(t('best_cost'), f"{res['best_cost']:,.0f}")
    colB.metric(t('lp_val'), f"{res['lp_val']:,.0f}")
    colC.metric(t('gap'), f"{res['gap']:.4f}%")
    colD.metric(t('open_fac'), res['open_fac'])

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
        st.dataframe(df_cycles, use_container_width=True)

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

# قراءة معلومات المستودع من secrets
REPO_OWNER = st.secrets.get("REPO_OWNER", "zakibeny")
REPO_NAME = st.secrets.get("REPO_NAME", "resolve-ilp-integer-linear-programing-")
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")

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
```
