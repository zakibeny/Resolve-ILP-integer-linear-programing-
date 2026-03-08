# ------------------- قسم إرسال التعليقات إلى GitHub (معدل) -------------------
st.markdown("---")
st.header(t('feedback_section'))

# قراءة معلومات المستودع من secrets بشكل آمن
try:
    REPO_OWNER = st.secrets.get("REPO_OWNER", "zakibeny")
    REPO_NAME = st.secrets.get("REPO_NAME", "resolve-ilp-integer-linear-programing-")
    GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
except Exception:
    # في حالة عدم وجود secrets (بيئة محلية)
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
