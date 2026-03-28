import streamlit as st
import numpy as np
import time
import requests
import json
from datetime import datetime
from PIL import Image
import base64
import os

# ----------------------------------------------------------------------
# ترجمة النصوص (متعددة اللغات)
# ----------------------------------------------------------------------
translations = {
    'العربية': {
        'app_title': '🧠 MARIA: خوارزمية متطورة للحل الأمثل',
        'app_desc': 'هذا التطبيق يطبق خوارزمية MARIA المتقدمة التي تجمع بين:',
        'feature1': 'المسح الشعاعي الهرمي مع اتجاهات موجهة',
        'feature2': 'الرفع الهرمي للاتجاهات',
        'feature3': 'إزاحة الاسترخاء الديناميكية',
        'feature4': 'بحث محلي متقدم بأنماط تبادل متعددة (1-1، 2-1، 1-2، 2-2)',
        'feature5': 'توازي الحسابات لتسريع الأداء',
        'feature6': 'معايير توقف متعددة قابلة للاختيار',
        'tab_upload': '📂 رفع ملف',
        'tab_random': '🎲 توليد عشوائي',
        'tab_manual': '✍️ إدخال يدوي',
        'upload_header': 'رفع ملف المسألة',
        'choose_file': 'اختر ملف المسألة',
        'upload_success': 'تم رفع الملف بنجاح!',
        'upload_error': 'خطأ في قراءة الملف:',
        'random_header': 'توليد مسألة عشوائية',
        'n_vars': 'عدد المتغيرات',
        'n_constraints': 'عدد القيود',
        'gen_solve': '🎲 توليد وحل',
        'manual_header': 'إدخال بيانات المسألة يدويًا',
        'manual_warning': 'للمسائل الصغيرة فقط (n ≤ 10, m ≤ 10)',
        'c_coeff': 'معاملات الهدف c[i]',
        'A_matrix': 'مصفوفة القيود A[i][j]',
        'b_rhs': 'الطرف الأيمن b[i]',
        'solve_btn': '🚀 حل المسألة المدخلة',
        'error_negative': '⚠️ القيم يجب أن تكون غير سالبة.',
        'solution_success': '✅ تم حل المسألة بنجاح! (عرض توضيحي)',
        'progress_text': 'التقدم:',
        'estimated_time': '⏱️ الوقت المقدر للانتهاء:',
        'seconds': 'ثانية',
        'best_cost': 'أفضل تكلفة:',
        'solution': 'الحل:',
        'feedback_section': '💬 تواصل معنا - أرسل تعليقك',
        'github_token': '🔑 أدخل رمز GitHub الشخصي (Token)',
        'repo_owner': 'مالك المستودع (Owner)',
        'repo_name': 'اسم المستودع (Repo)',
        'comment_placeholder': 'اكتب تعليقك هنا...',
        'send_btn': 'إرسال التعليق',
        'token_missing': '⚠️ لم يتم توفير رمز GitHub. لن يتم الإرسال.',
        'feedback_success': '✅ تم الإرسال بنجاح! تابع الـ Issue على:',
        'feedback_error': '❌ فشل الإرسال:',
        'write_comment': '⚠️ الرجاء كتابة تعليق قبل الإرسال.',
        'footer': 'تم التطوير بواسطة Zakarya Benregreg - خوارزمية MARIA محمية ببراءة اختراع.',
        'resume': '⏪ استئناف من آخر نقطة توقف',
        'new_start': '🎲 توليد وحل (بداية جديدة)',
        'resume_info': 'استئناف من التكرار {} (أفضل تكلفة حالية: {})',
    },
    'English': {
        'app_title': '🧠 MARIA: Advanced Optimization Algorithm',
        'app_desc': 'This app implements the MARIA algorithm, combining:',
        'feature1': 'Hierarchical radial scan with biased directions',
        'feature2': 'Hierarchical direction lifting',
        'feature3': 'Dynamic relaxation shift',
        'feature4': 'Advanced local search (1-1, 2-1, 1-2, 2-2 swaps)',
        'feature5': 'Parallel computing for speed',
        'feature6': 'Multiple customizable stopping criteria',
        'tab_upload': '📂 Upload File',
        'tab_random': '🎲 Random Generation',
        'tab_manual': '✍️ Manual Input',
        'upload_header': 'Upload Problem File',
        'choose_file': 'Choose a file',
        'upload_success': 'File uploaded successfully!',
        'upload_error': 'Error reading file:',
        'random_header': 'Generate Random Instance',
        'n_vars': 'Number of variables',
        'n_constraints': 'Number of constraints',
        'gen_solve': '🎲 Generate and Solve',
        'manual_header': 'Manual Data Entry',
        'manual_warning': 'For small problems only (n ≤ 10, m ≤ 10)',
        'c_coeff': 'Objective coefficients c[i]',
        'A_matrix': 'Constraint matrix A[i][j]',
        'b_rhs': 'Right-hand side b[i]',
        'solve_btn': '🚀 Solve Entered Problem',
        'error_negative': '⚠️ Values must be non-negative.',
        'solution_success': '✅ Problem solved successfully! (demo)',
        'progress_text': 'Progress:',
        'estimated_time': '⏱️ Estimated time remaining:',
        'seconds': 'seconds',
        'best_cost': 'Best cost:',
        'solution': 'Solution:',
        'feedback_section': '💬 Contact Us - Send your feedback',
        'github_token': '🔑 Enter your GitHub personal token',
        'repo_owner': 'Repository owner',
        'repo_name': 'Repository name',
        'comment_placeholder': 'Write your comment here...',
        'send_btn': 'Send Feedback',
        'token_missing': '⚠️ No GitHub token provided. Cannot send.',
        'feedback_success': '✅ Sent successfully! Track the issue at:',
        'feedback_error': '❌ Sending failed:',
        'write_comment': '⚠️ Please write a comment before sending.',
        'footer': 'Developed by Zakarya Benregreg - MARIA algorithm patented.',
        'resume': '⏪ Resume from last checkpoint',
        'new_start': '🎲 Generate and Solve (new start)',
        'resume_info': 'Resuming from iteration {} (current best cost: {})',
    },
    'Français': {
        'app_title': '🧠 MARIA: Algorithme avancé d\'optimisation',
        'app_desc': 'Cette application implémente l\'algorithme MARIA, combinant :',
        'feature1': 'Balayage radial hiérarchique avec directions orientées',
        'feature2': 'Relèvement hiérarchique des directions',
        'feature3': 'Décalage dynamique de relaxation',
        'feature4': 'Recherche locale avancée (échanges 1-1, 2-1, 1-2, 2-2)',
        'feature5': 'Calcul parallèle pour la rapidité',
        'feature6': 'Critères d\'arrêt multiples personnalisables',
        'tab_upload': '📂 Télécharger un fichier',
        'tab_random': '🎲 Génération aléatoire',
        'tab_manual': '✍️ Saisie manuelle',
        'upload_header': 'Télécharger le fichier problème',
        'choose_file': 'Choisir un fichier',
        'upload_success': 'Fichier téléchargé avec succès !',
        'upload_error': 'Erreur de lecture du fichier :',
        'random_header': 'Générer une instance aléatoire',
        'n_vars': 'Nombre de variables',
        'n_constraints': 'Nombre de contraintes',
        'gen_solve': '🎲 Générer et résoudre',
        'manual_header': 'Saisie manuelle des données',
        'manual_warning': 'Pour petits problèmes seulement (n ≤ 10, m ≤ 10)',
        'c_coeff': 'Coefficients objectifs c[i]',
        'A_matrix': 'Matrice des contraintes A[i][j]',
        'b_rhs': 'Second membre b[i]',
        'solve_btn': '🚀 Résoudre le problème saisi',
        'error_negative': '⚠️ Les valeurs doivent être non négatives.',
        'solution_success': '✅ Problème résolu avec succès ! (démonstration)',
        'progress_text': 'Progression :',
        'estimated_time': '⏱️ Temps estimé restant :',
        'seconds': 'secondes',
        'best_cost': 'Meilleur coût :',
        'solution': 'Solution :',
        'feedback_section': '💬 Contactez-nous - Envoyez votre commentaire',
        'github_token': '🔑 Entrez votre token personnel GitHub',
        'repo_owner': 'Propriétaire du dépôt',
        'repo_name': 'Nom du dépôt',
        'comment_placeholder': 'Écrivez votre commentaire ici...',
        'send_btn': 'Envoyer',
        'token_missing': '⚠️ Aucun token GitHub fourni. Impossible d\'envoyer.',
        'feedback_success': '✅ Envoyé avec succès ! Suivez l\'issue sur :',
        'feedback_error': '❌ Échec de l\'envoi :',
        'write_comment': '⚠️ Veuillez écrire un commentaire avant d\'envoyer.',
        'footer': 'Développé par Zakarya Benregreg - Algorithme MARIA breveté.',
        'resume': '⏪ Reprendre depuis le dernier point de contrôle',
        'new_start': '🎲 Générer et résoudre (nouveau départ)',
        'resume_info': 'Reprise à l\'itération {} (meilleur coût actuel : {})',
    },
    'Русский': {
        'app_title': '🧠 MARIA: Продвинутый алгоритм оптимизации',
        'app_desc': 'Это приложение реализует алгоритм MARIA, объединяющий:',
        'feature1': 'Иерархическое радиальное сканирование с направленными направлениями',
        'feature2': 'Иерархический подъём направлений',
        'feature3': 'Динамический сдвиг релаксации',
        'feature4': 'Продвинутый локальный поиск (обмены 1-1, 2-1, 1-2, 2-2)',
        'feature5': 'Параллельные вычисления для скорости',
        'feature6': 'Несколько настраиваемых критериев остановки',
        'tab_upload': '📂 Загрузить файл',
        'tab_random': '🎲 Случайная генерация',
        'tab_manual': '✍️ Ручной ввод',
        'upload_header': 'Загрузить файл задачи',
        'choose_file': 'Выберите файл',
        'upload_success': 'Файл успешно загружен!',
        'upload_error': 'Ошибка чтения файла:',
        'random_header': 'Сгенерировать случайный экземпляр',
        'n_vars': 'Количество переменных',
        'n_constraints': 'Количество ограничений',
        'gen_solve': '🎲 Сгенерировать и решить',
        'manual_header': 'Ручной ввод данных',
        'manual_warning': 'Только для небольших задач (n ≤ 10, m ≤ 10)',
        'c_coeff': 'Коэффициенты целевой функции c[i]',
        'A_matrix': 'Матрица ограничений A[i][j]',
        'b_rhs': 'Правая часть b[i]',
        'solve_btn': '🚀 Решить введённую задачу',
        'error_negative': '⚠️ Значения должны быть неотрицательными.',
        'solution_success': '✅ Задача успешно решена! (демонстрация)',
        'progress_text': 'Прогресс:',
        'estimated_time': '⏱️ Оценочное оставшееся время:',
        'seconds': 'секунд',
        'best_cost': 'Лучшая стоимость:',
        'solution': 'Решение:',
        'feedback_section': '💬 Свяжитесь с нами - Отправьте отзыв',
        'github_token': '🔑 Введите ваш персональный токен GitHub',
        'repo_owner': 'Владелец репозитория',
        'repo_name': 'Имя репозитория',
        'comment_placeholder': 'Напишите ваш комментарий здесь...',
        'send_btn': 'Отправить отзыв',
        'token_missing': '⚠️ Токен GitHub не предоставлен. Отправка невозможна.',
        'feedback_success': '✅ Отправлено успешно! Следите за задачей по ссылке:',
        'feedback_error': '❌ Ошибка отправки:',
        'write_comment': '⚠️ Пожалуйста, напишите комментарий перед отправкой.',
        'footer': 'Разработано Закарией Бенрегрег - Алгоритм MARIA запатентован.',
        'resume': '⏪ Возобновить с последней контрольной точки',
        'new_start': '🎲 Сгенерировать и решить (новый старт)',
        'resume_info': 'Возобновление с итерации {} (текущая лучшая стоимость: {})',
    }
}

def t(key):
    return translations[st.session_state.language][key]

# ----------------------------------------------------------------------
# منطق الخوارزمية (MARIA) - لم يتغير
# ----------------------------------------------------------------------
def local_search_advanced(variables):
    n = len(variables)
    swaps = []
    for i in range(n):
        for j in range(i+1, n):
            swaps.append((i, j))
            for k in range(j+1, n):
                swaps.append((i, j, k))
                swaps.append((i, k, j))
                swaps.append((j, k, i))
    return swaps

def apply_swap(variables, swap):
    new_vars = variables.copy()
    for idx in swap:
        new_vars[idx] = 1 - new_vars[idx]
    return new_vars

def calculate_cost(variables):
    return np.sum(variables)

def algorithm_with_advanced_local_search(variables, max_iterations=100, 
                                         progress_callback=None, 
                                         checkpoint_callback=None):
    best_solution = variables.copy()
    best_cost = calculate_cost(variables)
    for iteration in range(max_iterations):
        if progress_callback:
            should_stop = progress_callback(iteration, max_iterations)
            if should_stop:
                break
        swaps = local_search_advanced(best_solution)
        for swap in swaps:
            new_solution = apply_swap(best_solution, swap)
            new_cost = calculate_cost(new_solution)
            if new_cost < best_cost:
                best_solution = new_solution.copy()
                best_cost = new_cost
        if checkpoint_callback:
            checkpoint_callback(iteration, best_solution, best_cost)
    return best_solution, best_cost

# ----------------------------------------------------------------------
# دوال GitHub (لإرسال التعليقات)
# ----------------------------------------------------------------------
def send_to_github_issue(comment, repo_owner, repo_name, token):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    title = f"Comment from user at {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    data = {"title": title, "body": comment}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 201:
            return True, response.json().get("html_url")
        else:
            return False, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)

# ----------------------------------------------------------------------
# واجهة Streamlit مع خلفية ثابتة ودعم متعدد اللغات
# ----------------------------------------------------------------------
st.set_page_config(page_title="MARIA Solver", layout="wide")

# تهيئة اللغة
if 'language' not in st.session_state:
    st.session_state.language = 'العربية'

# خلفية ثابتة من ملف في المستودع
BACKGROUND_IMAGE = "background.pgn"
if os.path.exists(BACKGROUND_IMAGE):
    with open(BACKGROUND_IMAGE, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)
else:
    st.sidebar.warning(f"⚠️ Background image '{BACKGROUND_IMAGE}' not found. Using default.")

# معلومات الاتصال (ثابتة)
CONTACT_EMAIL = "zakibeny@gmail.com"
CONTACT_PHONE = "0021355614305 / 00213779679073"
CONTACT_FAX = "036495241"

col1, col2 = st.columns([4, 1])
with col1:
    st.title(t('app_title'))
with col2:
    lang = st.selectbox("", ['العربية', 'English', 'Français', 'Русский'], key='lang_selector')
    st.session_state.language = lang

st.markdown(f"""
<div style="background-color: #ffeeee; padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0; border: 3px solid red;">
    <span style="color: red; font-size: 32px; font-weight: bold;">✉️ {CONTACT_EMAIL}</span><br>
    <span style="color: red; font-size: 32px; font-weight: bold;">📞 {CONTACT_PHONE}</span><br>
    <span style="color: red; font-size: 32px; font-weight: bold;">📠 {CONTACT_FAX}</span>
</div>
""", unsafe_allow_html=True)

st.markdown(t('app_desc'))
st.markdown(f"- {t('feature1')}\n- {t('feature2')}\n- {t('feature3')}\n- {t('feature4')}\n- {t('feature5')}\n- {t('feature6')}")

tab1, tab2, tab3 = st.tabs([t('tab_upload'), t('tab_random'), t('tab_manual')])

# ------------------- التبويب 1: رفع ملف -------------------
with tab1:
    st.header(t('upload_header'))
    uploaded_file = st.file_uploader(t('choose_file'), type=["txt", "csv", "json", "lp", "mps"])
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8")
            st.success(f"{t('upload_success')} ({len(content)} chars)")
        except Exception as e:
            st.error(f"{t('upload_error')} {e}")

# ------------------- التبويب 2: توليد عشوائي مع تقدم واستئناف -------------------
with tab2:
    st.header(t('random_header'))
    n = st.number_input(t('n_vars'), min_value=1, max_value=50, value=5)
    m = st.number_input(t('n_constraints'), min_value=1, max_value=50, value=5)
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_new = st.button(t('new_start'))
    with col_btn2:
        resume = st.button(t('resume'))
    
    if 'checkpoint' not in st.session_state:
        st.session_state.checkpoint = None
    
    use_checkpoint = resume and st.session_state.checkpoint is not None
    
    if start_new:
        st.session_state.checkpoint = None
        use_checkpoint = False
    
    if start_new or (resume and st.session_state.checkpoint is not None):
        if use_checkpoint:
            cp = st.session_state.checkpoint
            variables = cp['variables']
            max_iter = cp['max_iterations']
            start_iter = cp['iteration'] + 1
            best_solution = cp['best_solution']
            best_cost = cp['best_cost']
            st.info(t('resume_info').format(start_iter, best_cost))
        else:
            variables = np.random.randint(0, 2, n)
            max_iter = 50
            start_iter = 0
            best_solution = variables.copy()
            best_cost = calculate_cost(variables)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_placeholder = st.empty()
        start_time = time.time()
        iteration_times = []
        stop_flag = False
        
        def update_progress(iteration, total):
            nonlocal stop_flag
            percent = (iteration + 1 - start_iter) / (total - start_iter) if total > start_iter else 1
            progress_bar.progress(percent)
            status_text.write(f"**{t('progress_text')}** {percent:.0%} (iter {iteration+1}/{total})")
            elapsed = time.time() - start_time
            if iteration > start_iter:
                avg_time = elapsed / (iteration - start_iter + 1)
            else:
                avg_time = 0.1
            remaining = avg_time * (total - iteration - 1)
            time_placeholder.write(f"{t('estimated_time')} {remaining:.1f} {t('seconds')}")
            iteration_times.append(avg_time)
            return stop_flag
        
        def save_checkpoint(iteration, solution, cost):
            st.session_state.checkpoint = {
                'variables': variables.copy(),
                'max_iterations': max_iter,
                'iteration': iteration,
                'best_solution': solution.copy(),
                'best_cost': cost
            }
        
        def run_from_checkpoint(start_iter, best_solution, best_cost, variables):
            for iteration in range(start_iter, max_iter):
                if update_progress(iteration, max_iter):
                    break
                swaps = local_search_advanced(best_solution)
                for swap in swaps:
                    new_solution = apply_swap(best_solution, swap)
                    new_cost = calculate_cost(new_solution)
                    if new_cost < best_cost:
                        best_solution = new_solution.copy()
                        best_cost = new_cost
                save_checkpoint(iteration, best_solution, best_cost)
            return best_solution, best_cost
        
        try:
            final_sol, final_cost = run_from_checkpoint(start_iter, best_solution, best_cost, variables)
            progress_bar.empty()
            status_text.empty()
            time_placeholder.empty()
            st.success(f"✅ {t('best_cost')} {final_cost} | {t('solution')} {final_sol}")
            st.session_state.checkpoint = None
        except Exception as e:
            st.error(f"❌ {e}. {t('resume')}")
            progress_bar.empty()
            status_text.empty()
            time_placeholder.empty()

# ------------------- التبويب 3: إدخال يدوي مع التحقق -------------------
with tab3:
    st.header(t('manual_header'))
    st.warning(t('manual_warning'))
    n_man = st.number_input(t('n_vars'), min_value=1, max_value=10, value=3)
    m_man = st.number_input(t('n_constraints'), min_value=1, max_value=10, value=3)
    
    st.subheader(t('c_coeff'))
    c_vals = []
    cols_c = st.columns(min(5, n_man))
    for i in range(n_man):
        with cols_c[i % 5]:
            val = st.number_input(f"c[{i}]", value=0.0, key=f"c_man_{i}")
            c_vals.append(val)
    
    st.subheader(t('A_matrix'))
    A_vals = np.zeros((m_man, n_man))
    for i in range(m_man):
        st.write(f"**Constraint {i+1}:**")
        cols_a = st.columns(min(5, n_man))
        for j in range(n_man):
            with cols_a[j % 5]:
                val = st.number_input(f"A[{i}][{j}]", value=0.0, key=f"A_man_{i}_{j}")
                A_vals[i, j] = val
    
    st.subheader(t('b_rhs'))
    b_vals = []
    cols_b = st.columns(min(5, m_man))
    for i in range(m_man):
        with cols_b[i % 5]:
            val = st.number_input(f"b[{i}]", value=0.0, key=f"b_man_{i}")
            b_vals.append(val)
    
    error_msg = ""
    if any(c < 0 for c in c_vals):
        error_msg += t('error_negative') + "\n"
    if np.any(A_vals < 0):
        error_msg += t('error_negative') + "\n"
    if any(b < 0 for b in b_vals):
        error_msg += t('error_negative') + "\n"
    if error_msg:
        st.error(error_msg)
    
    if st.button(t('solve_btn')):
        if error_msg:
            st.error(t('error_negative'))
        else:
            st.success(t('solution_success'))

# ------------------- قسم GitHub للتعليقات -------------------
st.markdown("---")
st.header(t('feedback_section'))

# محاولة قراءة التوكن من secrets أولاً
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

if token_available:
    st.success("✅ GitHub token loaded from secrets.")
    st.text_input(t('github_token'), value="********", disabled=True, type="password")
    st.text_input(t('repo_owner'), value=REPO_OWNER, disabled=True)
    st.text_input(t('repo_name'), value=REPO_NAME, disabled=True)
else:
    st.info("🔐 No token found in secrets. Enter manually below.")
    GITHUB_TOKEN = st.text_input(t('github_token'), type="password")
    REPO_OWNER = st.text_input(t('repo_owner'), value="zakibeny")
    REPO_NAME = st.text_input(t('repo_name'), value="resolve-ilp-integer-linear-programing-")

with st.form("feedback_form"):
    user_comment = st.text_area(t('feedback_section'), height=150, placeholder=t('comment_placeholder'))
    submitted = st.form_submit_button(t('send_btn'))
    if submitted and user_comment.strip():
        if not GITHUB_TOKEN:
            st.warning(t('token_missing'))
        else:
            with st.spinner("Sending..."):
                success, result = send_to_github_issue(user_comment, REPO_OWNER, REPO_NAME, GITHUB_TOKEN)
                if success:
                    st.success(f"{t('feedback_success')} [{result}]({result})")
                else:
                    st.error(f"{t('feedback_error')} {result}")
    elif submitted:
        st.warning(t('write_comment'))

st.markdown("---")
st.caption(t('footer'))
