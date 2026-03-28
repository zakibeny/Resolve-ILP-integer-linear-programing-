import streamlit as st
import numpy as np
import time
import requests
import json
from datetime import datetime
from PIL import Image
import base64

# ----------------------------------------------------------------------
# منطق الخوارزمية (بدون تغيير جوهري)
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
    """
    نفس الخوارزمية مع إمكانية حفظ نقاط توقف.
    checkpoint_callback يتلقى (iteration, best_solution, best_cost)
    """
    best_solution = variables.copy()
    best_cost = calculate_cost(variables)
    for iteration in range(max_iterations):
        if progress_callback:
            should_stop = progress_callback(iteration, max_iterations)
            if should_stop:  # يمكن إيقاف يدوي
                break
        # البحث المحلي
        swaps = local_search_advanced(best_solution)
        for swap in swaps:
            new_solution = apply_swap(best_solution, swap)
            new_cost = calculate_cost(new_solution)
            if new_cost < best_cost:
                best_solution = new_solution.copy()
                best_cost = new_cost
        # حفظ نقطة توقف
        if checkpoint_callback:
            checkpoint_callback(iteration, best_solution, best_cost)
    return best_solution, best_cost

# ----------------------------------------------------------------------
# دوال GitHub
# ----------------------------------------------------------------------
def send_to_github_issue(comment, repo_owner, repo_name, token):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    title = f"تعليق من مستخدم في {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    data = {"title": title, "body": comment}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 201:
            return True, response.json().get("html_url")
        else:
            return False, f"خطأ {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)

# ----------------------------------------------------------------------
# واجهة Streamlit
# ----------------------------------------------------------------------
st.set_page_config(page_title="AHRH Solver", layout="wide")

CONTACT_EMAIL = "zakibeny@gmail.com"
CONTACT_PHONE = "0021355614305 / 00213779679073"
CONTACT_FAX = "036495241"

col1, col2 = st.columns([4, 1])
with col1:
    st.title("🧠 MARIA: خوارزمية هرمية انكماشية متطورة")
with col2:
    language = st.selectbox("", ['English', 'Français', 'العربية', 'Русский'], key='lang')
    st.session_state.language = language

# خلفية
bg_file = st.sidebar.file_uploader("🌄 اختر صورة خلفية", type=["jpg", "png", "jpeg"])
if bg_file is not None:
    img_base64 = base64.b64encode(bg_file.getvalue()).decode()
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

st.markdown(f"""
<div style="background-color: #ffeeee; padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0; border: 3px solid red;">
    <span style="color: red; font-size: 32px; font-weight: bold;">✉️ {CONTACT_EMAIL}</span><br>
    <span style="color: red; font-size: 32px; font-weight: bold;">📞 {CONTACT_PHONE}</span><br>
    <span style="color: red; font-size: 32px; font-weight: bold;">📠 {CONTACT_FAX}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
هذا التطبيق يطبق خوارزمية AHRH المتقدمة التي تجمع بين:
- المسح الشعاعي الهرمي مع اتجاهات موجهة
- الرفع الهرمي للاتجاهات
- إزاحة الاسترخاء الديناميكية
- بحث محلي متقدم بأنماط تبادل متعددة (1-1، 2-1، 1-2، 2-2)
- توازي الحسابات لتسريع الأداء
- معايير توقف متعددة قابلة للاختيار
""")

tab1, tab2, tab3 = st.tabs(["📂 رفع ملف", "🎲 توليد عشوائي", "✍️ إدخال يدوي"])

# ------------------- التبويب 2: توليد عشوائي مع تقدم واستئناف -------------------
with tab2:
    st.header("توليد مسألة عشوائية")
    n = st.number_input("عدد المتغيرات", min_value=1, max_value=50, value=5)
    m = st.number_input("عدد القيود", min_value=1, max_value=50, value=5)
    
    # أزرار التشغيل والاستئناف
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_new = st.button("🎲 توليد وحل (بداية جديدة)")
    with col_btn2:
        resume = st.button("⏪ استئناف من آخر نقطة توقف")
    
    # تهيئة حالة الجلسة للتخزين
    if 'checkpoint' not in st.session_state:
        st.session_state.checkpoint = None
    
    # متغير لتحديد ما إذا كنا نستأنف
    use_checkpoint = resume and st.session_state.checkpoint is not None
    
    if start_new:
        st.session_state.checkpoint = None
        use_checkpoint = False
    
    if start_new or (resume and st.session_state.checkpoint is not None):
        # تحضير المتغيرات
        if use_checkpoint:
            cp = st.session_state.checkpoint
            variables = cp['variables']
            max_iter = cp['max_iterations']
            start_iter = cp['iteration'] + 1
            best_solution = cp['best_solution']
            best_cost = cp['best_cost']
            st.info(f"استئناف من التكرار {start_iter} (أفضل تكلفة حالية: {best_cost})")
        else:
            variables = np.random.randint(0, 2, n)
            max_iter = 50  # يمكن جعلها قابلة للتعديل
            start_iter = 0
            best_solution = variables.copy()
            best_cost = calculate_cost(variables)
        
        # عناصر واجهة التقدم
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_placeholder = st.empty()
        start_time = time.time()
        iteration_times = []
        stop_flag = False  # للإيقاف اليدوي (يمكن إضافة زر إيقاف)
        
        def update_progress(iteration, total):
            nonlocal stop_flag
            # إذا كان هناك زر إيقاف في المستقبل يمكن تفعيله
            # stop_flag = st.session_state.get('stop', False)
            percent = (iteration + 1 - start_iter) / (total - start_iter) if total > start_iter else 1
            progress_bar.progress(percent)
            status_text.write(f"**التقدم:** {percent:.0%} (التكرار {iteration+1}/{total})")
            elapsed = time.time() - start_time
            if iteration > start_iter:
                avg_time = elapsed / (iteration - start_iter + 1)
            else:
                avg_time = 0.1
            remaining = avg_time * (total - iteration - 1)
            time_placeholder.write(f"⏱️ الوقت المقدر للانتهاء: {remaining:.1f} ثانية")
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
        
        # تشغيل الخوارزمية مع تقدم ونقاط توقف
        # نمرر القيم الأولية إذا استأنفنا
        # نحتاج لتعديل الخوارزمية لتبدأ من نقطة معينة
        # نصنع دالة خاصة لهذا المثال:
        def run_from_checkpoint(start_iter, best_solution, best_cost, variables):
            for iteration in range(start_iter, max_iter):
                if update_progress(iteration, max_iter):
                    break
                # البحث المحلي
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
            st.success(f"✅ أفضل تكلفة: {final_cost} | الحل: {final_sol}")
            # مسح نقطة التوقف بعد الانتهاء بنجاح
            st.session_state.checkpoint = None
        except Exception as e:
            st.error(f"❌ حدث خطأ: {e}. تم حفظ آخر نقطة توقف، يمكنك استئناف لاحقاً.")
            # checkpoint محفوظ بالفعل
            progress_bar.empty()
            status_text.empty()
            time_placeholder.empty()

# ------------------- التبويب 1 و 3 (كما هي) -------------------
with tab1:
    st.header("رفع ملف المسألة")
    uploaded_file = st.file_uploader("اختر ملف المسألة", type=["txt", "csv", "json", "lp", "mps"])
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8")
            st.success(f"تم رفع الملف بنجاح! (حجمه {len(content)} حرف)")
        except Exception as e:
            st.error(f"خطأ في قراءة الملف: {e}")

with tab3:
    st.header("إدخال بيانات المسألة يدويًا")
    st.warning("للمسائل الصغيرة فقط (n ≤ 10, m ≤ 10)")
    n_man = st.number_input("عدد المتغيرات (n)", min_value=1, max_value=10, value=3)
    m_man = st.number_input("عدد القيود (m)", min_value=1, max_value=10, value=3)
    
    st.subheader("معاملات الهدف c[i]")
    c_vals = []
    cols_c = st.columns(min(5, n_man))
    for i in range(n_man):
        with cols_c[i % 5]:
            val = st.number_input(f"c[{i}]", value=0.0, key=f"c_man_{i}")
            c_vals.append(val)
    
    st.subheader("مصفوفة القيود A[i][j]")
    A_vals = np.zeros((m_man, n_man))
    for i in range(m_man):
        st.write(f"**القيود {i+1}:**")
        cols_a = st.columns(min(5, n_man))
        for j in range(n_man):
            with cols_a[j % 5]:
                val = st.number_input(f"A[{i}][{j}]", value=0.0, key=f"A_man_{i}_{j}")
                A_vals[i, j] = val
    
    st.subheader("الطرف الأيمن b[i]")
    b_vals = []
    cols_b = st.columns(min(5, m_man))
    for i in range(m_man):
        with cols_b[i % 5]:
            val = st.number_input(f"b[{i}]", value=0.0, key=f"b_man_{i}")
            b_vals.append(val)
    
    error_msg = ""
    if any(c < 0 for c in c_vals):
        error_msg += "⚠️ معاملات الهدف يجب أن تكون غير سالبة.\n"
    if np.any(A_vals < 0):
        error_msg += "⚠️ عناصر المصفوفة A يجب أن تكون غير سالبة.\n"
    if any(b < 0 for b in b_vals):
        error_msg += "⚠️ الطرف الأيمن b يجب أن يكون غير سالب.\n"
    if error_msg:
        st.error(error_msg)
    
    if st.button("🚀 حل المسألة المدخلة"):
        if error_msg:
            st.error("❌ لا يمكن الحل بسبب أخطاء في الإدخال. راجع البيانات.")
        else:
            st.success("✅ تم حل المسألة بنجاح! (عرض توضيحي)")

# ------------------- قسم GitHub للتعليقات -------------------
st.markdown("---")
st.header("💬 تواصل معنا - أرسل تعليقك")

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
    st.success("✅ تم تحميل رمز GitHub من الإعدادات السرية.")
    st.text_input("🔑 رمز GitHub (مخفي)", value="********", disabled=True, type="password")
    st.text_input("مالك المستودع", value=REPO_OWNER, disabled=True)
    st.text_input("اسم المستودع", value=REPO_NAME, disabled=True)
else:
    st.info("🔐 لم يتم العثور على توكن في secrets. يمكنك إدخال بياناتك يدوياً أدناه.")
    GITHUB_TOKEN = st.text_input("🔑 أدخل رمز GitHub الشخصي (Token)", type="password")
    REPO_OWNER = st.text_input("مالك المستودع (Owner)", value="zakibeny")
    REPO_NAME = st.text_input("اسم المستودع (Repo)", value="resolve-ilp-integer-linear-programing-")

with st.form("feedback_form"):
    user_comment = st.text_area("تعليقك", height=150, placeholder="اكتب تعليقك هنا...")
    submitted = st.form_submit_button("إرسال التعليق")
    if submitted and user_comment.strip():
        if not GITHUB_TOKEN:
            st.warning("⚠️ لم يتم توفير رمز GitHub. لن يتم الإرسال.")
        else:
            with st.spinner("جاري الإرسال..."):
                success, result = send_to_github_issue(user_comment, REPO_OWNER, REPO_NAME, GITHUB_TOKEN)
                if success:
                    st.success(f"✅ تم الإرسال بنجاح! تابع الـ Issue على: {result}")
                else:
                    st.error(f"❌ فشل الإرسال: {result}")
    elif submitted:
        st.warning("⚠️ الرجاء كتابة تعليق قبل الإرسال.")

st.markdown("---")
st.caption("تم التطوير بواسطة Zakarya Benregreg - خوارزمية MARIA محمية ببراءة اختراع.")