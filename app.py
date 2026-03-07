# -*- coding: utf-8 -*-
"""
تطبيق AHRH الكامل: حل مسائل UFLP والمسائل الخطية العامة (LP/ILP)
مع واجهة متعددة اللغات، عرض التقدم، معايير توقف متعددة، وإرسال التعليقات إلى GitHub Issues
(نسخة واحدة متكاملة - جميع الميزات السابقة + إضافة الإرسال)
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

# ------------------- معلومات الاتصال (تظهر في أعلى الصفحة) -------------------
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
        'problem_type': 'نوع المسألة',
        'lp': 'برمجة خطية مستمرة (LP)',
        'ilp': 'برمجة خطية صحيحة (ILP)',
        'file_type': 'نوع الملف',
        'uflp': 'ملف UFLP (مسألة مواقع المرافق)',
        'general': 'ملف عام (دالة هدف + قيود)',
        'file_format_help': '📄 شرح تنسيق الملفات',
        'uflp_format': """
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
""",
        'general_format': """
**تنسيق الملف العام (لأي مسألة LP/ILP):**
- السطر الأول: `n m` حيث n عدد المتغيرات، m عدد القيود.
- السطر الثاني: قيم معاملات الهدف `c` (n قيمة مفصولة بمسافات).
- ثم m سطر، كل سطر يمثل قيود `A` (n قيمة لكل سطر).
- السطر الأخير: قيم الطرف الأيمن `b` (m قيمة مفصولة بمسافات).
- مثال (مسألة صغيرة):
```

2 1
1 2
1 2
3

```
  تمثل: min x + 2y, subject to x + 2y <= 3, x,y >=0
""",
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
        'upload_info': 'يدعم أي ملف نصي (txt, dat, bub, opt, ...). يتم تجاهل الأسطر التي تبدأ بـ # أو ! أو FILE:',
        'choose_file': 'اختر ملف المسألة',
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
        'lp': 'Linear Programming (LP)',
        'ilp': 'Integer Linear Programming (ILP)',
        'file_type': 'File Type',
        'uflp': 'UFLP file (Facility Location)',
        'general': 'General file (Objective + Constraints)',
        'file_format_help': '📄 File Format Help',
        'uflp_format': """
**UFLP File Format (e.g., gs250, capb):**
- First line: `n m 0` where n = number of facilities, m = number of customers.
- Then n lines, each representing a facility: `[facility index] [opening cost] [transport cost to customer 1] [transport cost to customer 2] ... [transport cost to customer m]`
- Example:
```

5 3 0
1 100 5 6 7
2 150 8 9 10
3 120 4 5 6
4 130 7 8 9
5 110 3 4 5

```
- Lines starting with #, !, or FILE: are ignored.
""",
        'general_format': """
**General File Format (for any LP/ILP problem):**
- First line: `n m` where n = number of variables, m = number of constraints.
- Second line: objective coefficients `c` (n space-separated values).
- Then m lines, each representing a constraint `A` (n values per line).
- Last line: right-hand side `b` (m space-separated values).
- Example (small problem):
```

2 1
1 2
1 2
3

```
  Represents: min x + 2y, subject to x + 2y <= 3, x,y >=0
""",
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
        'upload_info': 'Accepts any text file (txt, dat, bub, opt, ...). Lines starting with #, !, or FILE: are ignored.',
        'choose_file': 'Choose a file',
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
        'lp': 'Programmation linéaire (LP)',
        'ilp': 'Programmation linéaire en nombres entiers (ILP)',
        'file_type': 'Type de fichier',
        'uflp': 'Fichier UFLP (Localisation d\'installations)',
        'general': 'Fichier général (Objectif + Contraintes)',
        'file_format_help': '📄 Aide sur le format des fichiers',
        'uflp_format': """
**Format du fichier UFLP (ex. gs250, capb):**
- Première ligne : `n m 0` où n = nombre de sites, m = nombre de clients.
- Ensuite n lignes, chacune représentant un site : `[indice du site] [coût d\'ouverture] [coût de transport vers client 1] [coût de transport vers client 2] ... [coût de transport vers client m]`
- Exemple :
```

5 3 0
1 100 5 6 7
2 150 8 9 10
3 120 4 5 6
4 130 7 8 9
5 110 3 4 5

```
- Les lignes commençant par #, ! ou FILE: sont ignorées.
""",
        'general_format': """
**Format du fichier général (pour tout problème LP/ILP):**
- Première ligne : `n m` où n = nombre de variables, m = nombre de contraintes.
- Deuxième ligne : coefficients objectifs `c` (n valeurs séparées par des espaces).
- Ensuite m lignes, chacune représentant une contrainte `A` (n valeurs par ligne).
- Dernière ligne : second membre `b` (m valeurs séparées par des espaces).
- Exemple (petit problème) :
```

2 1
1 2
1 2
3

```
  Représente : min x + 2y, sujet à x + 2y <= 3, x,y >=0
""",
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
        'upload_info': 'Accepte tout fichier texte (txt, dat, bub, opt, ...). Les lignes commençant par #, ! ou FILE: sont ignorées.',
        'choose_file': 'Choisir un fichier',
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
                y_cand_int 
