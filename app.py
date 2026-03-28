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
import base64
import os
warnings.filterwarnings("ignore")

# ------------------- معلومات الاتصال -------------------
CONTACT_EMAIL = "zakibeny@gmail.com"
CONTACT_PHONE = "0021355614305 / 00213779679073"
CONTACT_FAX = "036495241"

# ------------------- إعدادات التوازي -------------------
NUM_WORKERS = 4

# ------------------- ترجمة النصوص -------------------
translations = {
    'العربية': {
        'app_title': '🧠 MARIA: خوارزمية البرمجة الخطية الصحيحة ILP UFPS mbs.transport',
        'app_desc': 'هذا التطبيق يطبق خوارزمية MARIA المتقدمة التي تجمع بين:',
        'feature1': 'المسح الشعاعي الهرمي مع اتجاهات موجهة',
        'feature2': 'الرفع الهرمي للاتجاهات',
        'feature3': 'إزاحة الاسترخاء الديناميكية',
        'feature4': 'بحث محلي متقدم بأنماط تبادل متعددة (1-1، 2-1، 1-2، 2-2)',
        'feature5': 'توازي الحسابات لتسريع الأداء',
        'feature6': 'معايير توقف متعددة قابلة للاختيار',
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
        'footer': 'تم التطوير بواسطة Zakarya Benregreg - خوارزمية MARIA محمية ببراءة اختراع.',
        'acceleration_on': '⚡ وضع التسريع مُفعّل (فجوة < 2% و R < 0.01)',
        'acceleration_off': 'وضع التسريع غير مُفعّل',
        'feedback_section': '💬 تواصل معنا - أرسل تعليقك',
        'feedback_placeholder': 'اكتب تعليقك هنا... (سيتم إرساله كـ Issue في GitHub)',
        'feedback_submit': 'إرسال التعليق',
        'feedback_success': '✅ تم الإرسال بنجاح! يمكنك متابعة الـ issue على الرابط:',
        'feedback_error': '❌ فشل الإرسال:',
        'feedback_missing_token': '⚠️ خدمة إرسال التعليقات غير مفعلة حالياً.',
        'feedback_warning': 'الرجاء كتابة تعليق قبل الإرسال.',
        'progress_text': 'التقدم:',
        'estimated_time': 'الوقت المقدر للانتهاء:',
        'seconds': 'ثانية',
        'processing_file': 'جاري معالجة الملف وتشغيل الخوارزمية...',
        'file_processed': 'تمت معالجة الملف وتطبيق الخوارزمية.',
        'constraint_label': 'القيود',
        'upload_error': 'خطأ في قراءة الملف:',
        'upload_success': 'تم رفع الملف بنجاح!',
    },
    'English': {
        'app_title': '🧠 MARIA: Integer Linear Programming ILP UFPS mbs.transport',
        'app_desc': 'This app implements the MARIA algorithm, combining:',
        'feature1': 'Hierarchical radial scan with biased directions',
        'feature2': 'Hierarchical direction lifting',
        'feature3': 'Dynamic relaxation shift',
        'feature4': 'Advanced local search (1-1, 2-1, 1-2, 2-2 swaps)',
        'feature5': 'Parallel computing for speed',
        'feature6': 'Multiple customizable stopping criteria',
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
        'footer': 'Developed by Zakarya Benregreg - MARIA algorithm patented.',
        'acceleration_on': '⚡ Acceleration mode ON (gap < 2% and R < 0.01)',
        'acceleration_off': 'Acceleration mode OFF',
        'feedback_section': '💬 Contact Us - Send your feedback',
        'feedback_placeholder': 'Write your comment here... (will be sent as a GitHub Issue)',
        'feedback_submit': 'Send Feedback',
        'feedback_success': '✅ Sent successfully! You can track the issue at:',
        'feedback_error': '❌ Sending failed:',
        'feedback_missing_token': '⚠️ Feedback service is currently disabled.',
        'feedback_warning': 'Please write a comment before sending.',
        'progress_text': 'Progress:',
        'estimated_time': 'Estimated time remaining:',
        'seconds': 'seconds',
        'processing_file': 'Processing file and running algorithm...',
        'file_processed': 'File processed and algorithm applied.',
        'constraint_label': 'Constraint',
        'upload_error': 'Error reading file:',
        'upload_success': 'File uploaded successfully!',
    },
    'Français': {
        'app_title': '🧠 MARIA: Programmation Linéaire en Nombres Entiers ILP UFPS mbs.transport',
        'app_desc': 'Cette application implémente l\'algorithme MARIA, combinant :',
        'feature1': 'Balayage radial hiérarchique avec directions orientées',
        'feature2': 'Relèvement hiérarchique des directions',
        'feature3': 'Décalage dynamique de relaxation',
        'feature4': 'Recherche locale avancée (échanges 1-1, 2-1, 1-2, 2-2)',
        'feature5': 'Calcul parallèle pour la rapidité',
        'feature6': 'Critères d\'arrêt multiples personnalisables',
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
        'manual_header': 'Saisie manuelle des données',
        'manual_warning': 'Pour petits problèmes seulement (n ≤ 10, m ≤ 10)',
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
        'footer': 'Développé par Zakarya Benregreg - Algorithme MARIA breveté.',
        'acceleration_on': '⚡ Mode accélération ACTIVÉ (gap < 2% et R < 0.01)',
        'acceleration_off': 'Mode accélération DÉSACTIVÉ',
        'feedback_section': '💬 Contactez-nous - Envoyez votre commentaire',
        'feedback_placeholder': 'Écrivez votre commentaire ici... (sera envoyé comme Issue GitHub)',
        'feedback_submit': 'Envoyer',
        'feedback_success': '✅ Envoyé avec succès ! Suivez l\'issue sur :',
        'feedback_error': '❌ Échec de l\'envoi :',
        'feedback_missing_token': '⚠️ Le service de commentaires est actuellement désactivé.',
        'feedback_warning': 'Veuillez écrire un commentaire avant d\'envoyer.',
        'progress_text': 'Progression :',
        'estimated_time': 'Temps estimé restant :',
        'seconds': 'secondes',
        'processing_file': 'Traitement du fichier et exécution de l\'algorithme...',
        'file_processed': 'Fichier traité et algorithme appliqué.',
        'constraint_label': 'Contrainte',
        'upload_error': 'Erreur de lecture du fichier :',
        'upload_success': 'Fichier téléchargé avec succès !',
    },
    'Русский': {
        'app_title': '🧠 MARIA: Целочисленное линейное программирование ILP UFPS mbs.transport',
        'app_desc': 'Это приложение реализует алгоритм MARIA, объединяющий:',
        'feature1': 'Иерархическое радиальное сканирование с направленными направлениями',
        'feature2': 'Иерархический подъём направлений',
        'feature3': 'Динамический сдвиг релаксации',
        'feature4': 'Продвинутый локальный поиск (обмены 1-1, 2-1, 1-2, 2-2)',
        'feature5': 'Параллельные вычисления для скорости',
        'feature6': 'Несколько настраиваемых критериев остановки',
        'problem_type': 'Тип задачи',
        'ilp': 'Общее целочисленное программирование (ILP)',
        'uflp': 'Размещение объектов (UFLP)',
        'sidebar_algo': '⚙️ Параметры алгоритма',
        'max_cycles': 'Макс. циклов',
        'k_coarse': 'Размер грубого набора (k)',
        'patience': 'Терпение (циклов без улучшения)',
        'sidebar_stop': '⏹️ Критерии остановки',
        'choose_criteria': 'Выберите любую комбинацию (алгоритм останавливается при выполнении любого условия):',
        'use_R': 'Использовать порог R (с устойчивостью разрыва)',
        'R_tol': 'Допуск R (ε)',
        'stable_gap': 'Требуемые циклы устойчивости разрыва',
        'use_cost_repeat': 'Использовать повторение стоимости',
        'cost_repeat_times': 'Количество повторений',
        'use_gap_repeat': 'Использовать повторение разрыва',
        'gap_repeat_times': 'Количество повторений',
        'use_contraction': 'Использовать критерий сжатия (diff + R)',
        'diff_tol': 'Допуск разницы решений (ε₁)',
        'workers': 'Рабочие (параллельные потоки)',
        'tab_upload': '📂 Загрузить файл',
        'tab_manual': '✍️ Ручной ввод',
        'upload_header': 'Загрузить файл задачи',
        'upload_info': 'Поддерживает файлы UFLP (например, gs250) и общие файлы ILP.',
        'choose_file': 'Выберите файл',
        'file_type': 'Тип файла',
        'uflp_file': 'Файл UFLP (n m 0 + данные)',
        'ilp_file': 'Общий файл ILP (n m + c + A + b)',
        'ilp_format_help': """
**Формат общего файла ILP:**
- Первая строка: `n m` (количество переменных, количество ограничений)
- Вторая строка: коэффициенты цели c (n значений)
- Затем m строк: матрица ограничений A (n значений в строке)
- Последняя строка: правая часть b (m значений)
""",
        'uflp_format_help': """
**Формат файла UFLP:**
- Первая строка: `n m 0` (количество объектов, количество клиентов)
- Затем n строк: `индекс_объекта` `стоимость_открытия` `стоимости_транспорта до m клиентов`
""",
        'manual_header': 'Ручной ввод данных',
        'manual_warning': 'Только для небольших задач (n ≤ 10, m ≤ 10)',
        'manual_n': 'Количество переменных (n)',
        'manual_m': 'Количество ограничений (m)',
        'manual_c': 'Коэффициенты цели c[i]',
        'manual_A': 'Матрица ограничений A[i][j]',
        'manual_b': 'Правая часть b[i]',
        'solve_button': '🚀 Решить введённую задачу',
        'results': '📊 Результаты',
        'best_cost': 'Лучшая стоимость',
        'lp_val': 'Значение LP',
        'gap': 'Разрыв',
        'open_fac': 'Открытые объекты',
        'cycles_done': 'Выполнено циклов',
        'time': 'Время (с)',
        'size': 'Размер задачи',
        'stop_reason': 'Причина остановки',
        'gap_plot': '📈 Эволюция разрыва и R',
        'gap_label': 'Разрыв (%)',
        'R_label': 'R',
        'cycle_log': '📋 Журнал циклов',
        'cycle': 'Цикл',
        'cost': 'Стоимость',
        'improved': 'Улучшение?',
        'best_so_far': 'Лучший на данный момент',
        'yes': 'Да',
        'no': 'Нет',
        'download': '📥 Скачать эволюцию (CSV)',
        'info_placeholder': '👈 Выберите источник данных и нажмите кнопку запуска.',
        'footer': 'Разработано Закарией Бенрегрег - Алгоритм MARIA запатентован.',
        'acceleration_on': '⚡ Режим ускорения ВКЛЮЧЁН (разрыв < 2% и R < 0.01)',
        'acceleration_off': 'Режим ускорения ВЫКЛЮЧЁН',
        'feedback_section': '💬 Свяжитесь с нами - Отправьте отзыв',
        'feedback_placeholder': 'Напишите ваш комментарий здесь... (будет отправлен как Issue на GitHub)',
        'feedback_submit': 'Отправить отзыв',
        'feedback_success': '✅ Отправлено успешно! Следите за задачей по ссылке:',
        'feedback_error': '❌ Ошибка отправки:',
        'feedback_missing_token': '⚠️ Сервис отзывов в настоящее время отключён.',
        'feedback_warning': 'Пожалуйста, напишите комментарий перед отправкой.',
        'progress_text': 'Прогресс:',
        'estimated_time': 'Оценочное оставшееся время:',
        'seconds': 'секунд',
        'processing_file': 'Обработка файла и запуск алгоритма...',
        'file_processed': 'Файл обработан, алгоритм применён.',
        'constraint_label': 'Ограничение',
        'upload_error': 'Ошибка чтения файла:',
        'upload_success': 'Файл успешно загружен!',
    }
}

def t(key):
    """دالة ترجمة آمنة - تعيد المفتاح إذا لم يوجد"""
    lang = st.session_state.language
    if lang in translations and key in translations[lang]:
        return translations[lang][key]
    return key  # fallback

# ------------------- دوال الخوارزمية (مختصرة للاختصار) -------------------
# [هنا توضع دوال الخوارزمية كاملة كما في الكود السابق]
# نظراً لطول الكود، تم حذف الدوال هنا، ولكن يجب تضمينها كاملة كما في الرد السابق.
# في هذا المثال، سأكتفي بالإشارة إلى أنه يجب وضع الدوال الأصلية.

# ------------------- واجهة Streamlit -------------------
st.set_page_config(page_title="MARIA Solver", layout="wide")

# تهيئة اللغة (الإنجليزية افتراضياً)
if 'language' not in st.session_state:
    st.session_state.language = 'English'

# ------------------- CSS محسن -------------------
st.markdown("""
<style>
    /* تعتيم الخلفية */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.7);
        z-index: -1;
    }
    .stApp {
        background-color: transparent !important;
    }
    /* خلفية بيضاء شفافة مع حواف مستديرة */
    div[data-testid="stMarkdownContainer"], 
    .stMarkdown, 
    .stText, 
    .stNumberInput, 
    .stSelectbox, 
    .stTextInput, 
    .stTextArea, 
    .stButton button,
    .stAlert,
    .stSuccess,
    .stError,
    .stWarning,
    .stInfo,
    .stExpander,
    .stTabs,
    .stTab,
    .element-container,
    .stHeader,
    .stSubheader,
    .stCaption,
    .stForm {
        background-color: rgba(255, 255, 255, 0.85) !important;
        border-radius: 12px;
        padding: 8px 12px;
        margin: 6px 0;
        color: #FFFFFF !important;
        font-size: 18px;
        line-height: 1.5;
        backdrop-filter: blur(3px);
    }
    h1, h2, h3, h4, h5, h6 {
        background-color: rgba(255, 255, 255, 0.85) !important;
        border-radius: 12px;
        padding: 8px 12px;
        margin: 10px 0;
        color: #FFFFFF !important;
        font-weight: bold;
        font-size: 1.6em;
    }
    /* النصوص داخل العناصر */
    p, div, span, label, .stMarkdown p, .stText p {
        color: #FFFFFF !important;
        font-weight: bold;
        font-size: 18px;
    }
    input, textarea, select {
        background-color: white !important;
        color: black !important;
        font-weight: bold;
        font-size: 16px !important;
        border-radius: 8px;
    }
    .stButton button {
        background-color: #ffffffcc !important;
        color: black !important;
        font-weight: bold;
        border: 1px solid #aaa;
    }
    .stButton button:hover {
        background-color: white !important;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: rgba(255, 255, 255, 0.85) !important;
        color: white !important;
        font-weight: bold;
    }
    /* المربع الأحمر */
    .contact-box {
        background-color: #ffeeee !important;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        border: 3px solid red;
        color: red;
    }
    .contact-box span {
        color: red;
        font-size: 32px;
        font-weight: bold;
    }
    /* تقليل الهوامش العلوية والسفلية */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    /* تحسين ظهور الأزرار */
    .stButton {
        margin-top: 5px;
        margin-bottom: 5px;
    }
    /* اللغة في الأعلى مع الأعلام */
    .lang-selector {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    .lang-selector select {
        background-color: white;
        color: black;
        font-weight: bold;
        border-radius: 20px;
        padding: 5px 10px;
    }
</style>
""", unsafe_allow_html=True)

# خلفية ثابتة
BACKGROUND_IMAGE = "background.png"
if os.path.exists(BACKGROUND_IMAGE):
    with open(BACKGROUND_IMAGE, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()
    if BACKGROUND_IMAGE.endswith('.png'):
        mime = "image/png"
    elif BACKGROUND_IMAGE.endswith('.jpg') or BACKGROUND_IMAGE.endswith('.jpeg'):
        mime = "image/jpeg"
    else:
        mime = "image/png"
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:{mime};base64,{img_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-color: transparent;
    }}
    </style>
    """, unsafe_allow_html=True)
else:
    st.sidebar.warning(f"⚠️ لم يتم العثور على ملف الخلفية '{BACKGROUND_IMAGE}'. سيتم استخدام خلفية بيضاء.")

# ------------------- اختيار اللغة بالأعلام -------------------
# نضع اللغة في أعلى الصفحة قبل المربع الأحمر
lang_col1, lang_col2 = st.columns([4, 1])
with lang_col2:
    # استخدم selectbox مع الأعلام
    lang_options = {
        'English': '🇬🇧 English',
        'Français': '🇫🇷 Français',
        'العربية': '🇸🇦 العربية',
        'Русский': '🇷🇺 Русский'
    }
    selected_lang = st.selectbox(
        "",
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        key='lang_top'
    )
    st.session_state.language = selected_lang

# معلومات الاتصال (المربع الأحمر)
st.markdown(f"""
<div class="contact-box">
    <span>✉️ {CONTACT_EMAIL}</span><br>
    <span>📞 {CONTACT_PHONE}</span><br>
    <span>📠 {CONTACT_FAX}</span>
</div>
""", unsafe_allow_html=True)

# رأس الصفحة (العنوان)
st.title(t('app_title'))

# وصف الخوارزمية
st.markdown(t('app_desc'))
st.markdown(f"- {t('feature1')}\n- {t('feature2')}\n- {t('feature3')}\n- {t('feature4')}\n- {t('feature5')}\n- {t('feature6')}")

# شريط جانبي للمعاملات
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

# اختيار نوع المسألة
problem_type = st.radio(t('problem_type'), [t('ilp'), t('uflp')])
is_uflp = (problem_type == t('uflp'))

tab1, tab2 = st.tabs([t('tab_upload'), t('tab_manual')])

# ------------------- التبويب 1: رفع ملف -------------------
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
        with st.spinner(t('processing_file')):
            try:
                text = uploaded_file.getvalue().decode("utf-8")
            except:
                text = uploaded_file.getvalue().decode("latin-1")
            try:
                if is_uflp:
                    f, c, n, m = read_uflp_file(text)  # يجب أن تكون الدوال معرفة
                    st.success(f"File loaded: {n} facilities, {m} customers")
                    
                    # عناصر التقدم
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    time_placeholder = st.empty()
                    start_time = time.time()
                    
                    def progress_callback(cycle, total, best_cost, gap, R):
                        percent = (cycle + 1) / total
                        progress_bar.progress(percent)
                        status_text.write(f"**{t('progress_text')}** {percent:.0%} (Cycle {cycle+1}/{total})")
                        elapsed = time.time() - start_time
                        if cycle > 0:
                            avg_time = elapsed / (cycle + 1)
                            remaining = avg_time * (total - cycle - 1)
                            time_placeholder.write(f"{t('estimated_time')} {remaining:.1f} {t('seconds')}")
                    
                    result = solve_ahrh_uflp(
                        f, c,
                        max_cycles, k_coarse, patience,
                        use_R, R_tol, stable_gap_needed,
                        use_cost_repeat, cost_repeat_times,
                        use_gap_repeat, gap_repeat_times,
                        use_contraction, diff_tol,
                        progress_callback=progress_callback
                    )
                    progress_bar.empty()
                    status_text.empty()
                    time_placeholder.empty()
                    st.session_state['result'] = result
                    st.session_state['n'] = n
                    st.session_state['m'] = m
                else:
                    c, A, b, n, m = read_ilp_file(text)
                    st.success(f"File loaded: {n} variables, {m} constraints")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    time_placeholder = st.empty()
                    start_time = time.time()
                    
                    def progress_callback(cycle, total, best_cost, gap, R):
                        percent = (cycle + 1) / total
                        progress_bar.progress(percent)
                        status_text.write(f"**{t('progress_text')}** {percent:.0%} (Cycle {cycle+1}/{total})")
                        elapsed = time.time() - start_time
                        if cycle > 0:
                            avg_time = elapsed / (cycle + 1)
                            remaining = avg_time * (total - cycle - 1)
                            time_placeholder.write(f"{t('estimated_time')} {remaining:.1f} {t('seconds')}")
                    
                    result, x = solve_ahrh_general_ilp(
                        c, A, b,
                        max_cycles=max_cycles, k_coarse=k_coarse, patience=patience,
                        use_R=use_R, R_tol=R_tol, stable_gap_needed=stable_gap_needed,
                        use_cost_repeat=use_cost_repeat, cost_repeat_times=cost_repeat_times,
                        use_gap_repeat=use_gap_repeat, gap_repeat_times=gap_repeat_times,
                        use_contraction=use_contraction, diff_tol=diff_tol,
                        progress_callback=progress_callback
                    )
                    progress_bar.empty()
                    status_text.empty()
                    time_placeholder.empty()
                    st.session_state['result'] = result
                    st.session_state['n'] = n
                    st.session_state['m'] = m
            except Exception as e:
                st.error(f"{t('upload_error')} {str(e)}")

# ------------------- التبويب 2: إدخال يدوي -------------------
with tab2:
    st.header(t('manual_header'))
    st.warning(t('manual_warning'))
    # ... (كما هو في الكود السابق) ...

# عرض النتائج (كما هو)
# ...

# ------------------- قسم GitHub للتعليقات -------------------
st.markdown("---")
st.header(t('feedback_section'))

# قراءة التوكن من secrets (بدون عرض حقول)
try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    REPO_OWNER = st.secrets.get("REPO_OWNER", "zakibeny")
    REPO_NAME = st.secrets.get("REPO_NAME", "resolve-ilp-integer-linear-programing-")
    token_available = True
except:
    token_available = False
    GITHUB_TOKEN = ""
    REPO_OWNER = "zakibeny"
    REPO_NAME = "resolve-ilp-integer-linear-programing-"

with st.form("feedback_form"):
    user_comment = st.text_area("", height=150, placeholder=t('feedback_placeholder'))
    submitted = st.form_submit_button(t('feedback_submit'))
    if submitted and user_comment.strip():
        if not token_available:
            st.warning(t('feedback_missing_token'))
        else:
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
