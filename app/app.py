import sys
from pathlib import Path

# Add project root and src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
from PIL import Image

from greenlens.inference.disease_info import DiseaseDB
from greenlens.inference.gradcam import GradCAM
from greenlens.inference.predict import PlantDiseasePredictor


# Set Page Configuration
st.set_page_config(
    page_title="GreenLens | AI Plant Disease Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Translations Dictionary
I18N = {
    "ar": {
        "title": "GreenLens — التشخيص الذكي لأمراض النباتات",
        "sub": "نظام رؤية حاسوبية مدعوم بالذكاء الاصطناعي لتشخيص أمراض أوراق المحاصيل وتقديم الإرشادات الزراعية الفورية.",
        "lang_switch": "اللغة / Language",
        "model_stats": "معلومات النموذج",
        "model_acc": "دقة النموذج (Accuracy)",
        "model_f1": "Macro F1-Score",
        "model_latency": "زمن الاستجابة (Latency)",
        "conf_thresh": "حد الثقة الأدنى (Confidence Threshold)",
        "upload_tab": "📁 رفع صورة من جهازك",
        "camera_tab": "📷 التقاط صورة مباشرة",
        "samples_tab": "🧪 تجربة عينات سريعة",
        "upload_label": "اختر صورة واضحة لورقة نبات واحدة (JPG, PNG, WebP)",
        "camera_label": "وجه الكاميرا نحو ورقة النبات والتقط الصورة",
        "sample_prompt": "اضغط على أي عينة لتجربتها فوراً:",
        "analyze_btn": "🔍 بدء الفحص والتشخيص",
        "diagnosing": "جاري تحليل أنسجة الورقة ومطابقة الأعراض...",
        "results_header": "📋 تقرير التشخيص الزراعي",
        "plant": "نوع المحصول",
        "disease": "التشخيص الطبي",
        "confidence": "نسبة ثقة النموذج",
        "status": "الحالة العامة",
        "healthy_tag": "نبات سليم ومعافى ✅",
        "diseased_tag": "إصابة مرضية مرصودة ⚠️",
        "severity": "درجة الخطورة",
        "top_preds": "📊 أعلى التوقعات المحتملة",
        "about": "📖 نبذة عن المرض والمسبب",
        "cause": "المسبب الرئيسي للمرض",
        "symptoms": "🔍 الأعراض الميدانية الشائعة",
        "treatment": "💊 خطة العلاج والتصرف السريع",
        "prevention": "🛡️ التدابير الوقائية لحماية المحصول مستقبلاً",
        "uncertain_warning": "⚠️ نسبة الثقة منخفضة نسبياً. قد تكون إضاءة الصورة غير مناسبة أو أن المرض خارج نطاق الفئات المدربة. يُرجى التقاط صورة مقربة وواضحة لورقة واحدة.",
        "disclaimer_title": "إخلاء مسؤولية زراعي",
    },
    "en": {
        "title": "GreenLens — AI Plant Disease Diagnosis",
        "sub": "Deep learning-powered computer vision system for real-time crop disease detection and agronomic guidance.",
        "lang_switch": "Language / اللغة",
        "model_stats": "Model Statistics",
        "model_acc": "Model Accuracy",
        "model_f1": "Macro F1-Score",
        "model_latency": "Inference Latency",
        "conf_thresh": "Confidence Threshold",
        "upload_tab": "📁 Upload Image",
        "camera_tab": "📷 Take a Photo",
        "samples_tab": "🧪 Try Samples",
        "upload_label": "Choose a clear leaf image (JPG, PNG, WebP)",
        "camera_label": "Point your camera at a single plant leaf",
        "sample_prompt": "Click any sample image to test instantly:",
        "analyze_btn": "🔍 Analyze & Diagnose",
        "diagnosing": "Analyzing leaf patterns and matching diseases...",
        "results_header": "📋 Diagnostic & Agronomic Report",
        "plant": "Crop",
        "disease": "Diagnosis",
        "confidence": "Model Confidence",
        "status": "General Status",
        "healthy_tag": "Healthy Plant ✅",
        "diseased_tag": "Disease Detected ⚠️",
        "severity": "Severity Level",
        "top_preds": "📊 Top Predictions",
        "about": "📖 About the Condition & Pathogen",
        "cause": "Primary Pathogen / Cause",
        "symptoms": "🔍 Visual Field Symptoms",
        "treatment": "💊 Immediate Treatment Plan",
        "prevention": "🛡️ Preventive Measures & Crop Protection",
        "uncertain_warning": "⚠️ Prediction confidence is low. Image may be blurry or out-of-domain. Please try a closer, well-lit single-leaf photo.",
        "disclaimer_title": "Agricultural Disclaimer",
    },
}

# Pre-defined test samples from the dataset
SAMPLE_IMAGES = {
    "🍅 Tomato Bacterial Spot (طماطم - تبقع بكتيري)": ROOT / "data/raw/PlantVillage/plantvillage/PlantVillage/Tomato_Bacterial_spot/934f7497-94be-4557-8e25-6bc9b27ba7ba___UF.GRC_BS_Lab Leaf 0392.JPG",
    "🫑 Pepper Bell Healthy (فلفل - سليم)": ROOT / "data/raw/PlantVillage/plantvillage/PlantVillage/Pepper__bell___healthy/98327d43-c692-4922-82e3-a4e0db73ae40___JR_HL 8483.JPG",
    "🥔 Potato Early Blight (بطاطس - لفحة مبكرة)": ROOT / "data/raw/PlantVillage/plantvillage/PlantVillage/Potato___Early_blight/b1b72e9d-9213-4330-9efd-a60d0ee89606___RS_Early.B 8721.JPG",
    "🍅 Tomato Yellow Leaf Curl (طماطم - فيروس تجعد واصفرار)": ROOT / "data/raw/PlantVillage/plantvillage/PlantVillage/Tomato__Tomato_YellowLeaf__Curl_Virus/6e0e10b3-69f0-404e-839d-03213389aab0___UF.GRC_YLCV_Lab 01919.JPG",
}


@st.cache_resource(show_spinner="Loading GreenLens AI Model...")
def get_predictor():
    checkpoint_path = ROOT / "models" / "best_model.pth"
    return PlantDiseasePredictor(checkpoint_path=checkpoint_path, min_confidence=0.60)


@st.cache_resource
def get_disease_db():
    return DiseaseDB()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/plant-under-sun.png", width=70)
    st.markdown("### 🌿 **GreenLens AI**")

    # Language Toggle
    selected_lang = st.radio("اللغة / Language", ["العربية", "English"], horizontal=True)
    lang = "ar" if selected_lang == "العربية" else "en"
    t = I18N[lang]

    st.markdown("---")
    st.markdown(f"#### ⚙️ **{t['model_stats']}**")
    st.markdown(f"• **Architecture:** ResNet-18 (Pretrained)")
    st.markdown(f"• **{t['model_acc']}:** **99.56%**")
    st.markdown(f"• **{t['model_f1']}:** **0.9958**")
    st.markdown(f"• **{t['model_latency']}:** **1.5 ms** (GPU) / **16 ms** (CPU)")

    st.markdown("---")
    min_confidence = st.slider(t["conf_thresh"], min_value=0.40, max_value=0.90, value=0.60, step=0.05)


# ---------------------------------------------------------
# Apply RTL styling for Arabic
# ---------------------------------------------------------
if lang == "ar":
    st.markdown(
        """
        <style>
        .stApp { direction: rtl; text-align: right; }
        .stMetric label { direction: rtl; text-align: right; }
        .stMetric div { direction: rtl; text-align: right; }
        .stExpander header { direction: rtl; text-align: right; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.title(f"🌿 {t['title']}")
st.markdown(f"*{t['sub']}*")

db = get_disease_db()
st.info(f"ℹ️ **{t['disclaimer_title']}:** {db.disclaimer(lang)}")

# Initialize session state for image persistence
if "current_image" not in st.session_state:
    st.session_state["current_image"] = None
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None
if "gradcam_overlay" not in st.session_state:
    st.session_state["gradcam_overlay"] = None

# ---------------------------------------------------------
# Input Tabs
# ---------------------------------------------------------
tab_upload, tab_camera, tab_samples = st.tabs([t["upload_tab"], t["camera_tab"], t["samples_tab"]])

with tab_upload:
    uploaded_file = st.file_uploader(t["upload_label"], type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        if st.session_state.get("last_uploaded_name") != uploaded_file.name:
            st.session_state["current_image"] = Image.open(uploaded_file)
            st.session_state["last_uploaded_name"] = uploaded_file.name
            st.session_state["last_prediction"] = None
            st.session_state["gradcam_overlay"] = None

with tab_camera:
    enable_camera = st.checkbox("📷 " + t["camera_label"], value=False)
    if enable_camera:
        camera_file = st.camera_input("")
        if camera_file is not None:
            st.session_state["current_image"] = Image.open(camera_file)
            st.session_state["last_prediction"] = None
            st.session_state["gradcam_overlay"] = None

with tab_samples:
    st.markdown(f"**{t['sample_prompt']}**")
    cols = st.columns(len(SAMPLE_IMAGES))
    for col, (name, path) in zip(cols, SAMPLE_IMAGES.items()):
        with col:
            if path.exists():
                sample_img = Image.open(path)
                st.image(sample_img, caption=name.split("(")[0].strip(), use_container_width=True)
                if st.button(f"اختيار 📌" if lang == "ar" else "Select 📌", key=f"btn_{name}", use_container_width=True):
                    st.session_state["current_image"] = sample_img
                    st.session_state["selected_sample_name"] = name
                    st.session_state["last_prediction"] = None
                    st.session_state["gradcam_overlay"] = None
                    st.rerun()

active_image = st.session_state.get("current_image")

# ---------------------------------------------------------
# Image Preview & Analysis
# ---------------------------------------------------------
if active_image is not None:
    st.markdown("---")
    col_img, col_info = st.columns([1, 1.3], gap="large")

    with col_img:
        st.markdown(f"### 🖼️ ورقة النبات محل الفحص" if lang == "ar" else "### 🖼️ Inspected Leaf Image")
        enable_cam = st.toggle("🔥 تفعيل خريطة الانتباه الحرارية (Grad-CAM)" if lang == "ar" else "🔥 Enable Grad-CAM Heatmap", value=True)
        analyze_clicked = st.button(t["analyze_btn"], type="primary", use_container_width=True)

        if analyze_clicked:
            with st.spinner(t["diagnosing"]):
                predictor = get_predictor()
                predictor.min_confidence = min_confidence
                st.session_state["last_prediction"] = predictor.predict(active_image, top_k=3, lang=lang)
                if enable_cam:
                    cam = GradCAM(predictor.model)
                    _, overlay_img, _ = cam.generate(active_image)
                    st.session_state["gradcam_overlay"] = overlay_img
                else:
                    st.session_state["gradcam_overlay"] = None

        # Display images based on state
        overlay_img = st.session_state.get("gradcam_overlay")
        if enable_cam and overlay_img is not None:
            tab_orig, tab_cam = st.tabs(["🍃 الورقة الأصلية" if lang == "ar" else "🍃 Original", "🔥 بؤرة تركيز الذكاء الاصطناعي" if lang == "ar" else "🔥 Grad-CAM Heatmap"])
            with tab_orig:
                st.image(active_image, use_container_width=True)
            with tab_cam:
                st.image(overlay_img, caption="Grad-CAM Attention Map (المناطق الأكثر تأثيراً في قرار الموديل)", use_container_width=True)
        else:
            st.image(active_image, use_container_width=True)

    with col_info:
        result = st.session_state.get("last_prediction")

        if result is not None:
            best = result.best
            info = best.info

            st.markdown(f"### {t['results_header']}")

            # Status Banner
            if result.is_healthy:
                st.success(f"### ✅ {t['healthy_tag']}")
            else:
                sev = info.get("severity", "moderate")
                sev_label = info.get("severity_label", "-")
                if sev in ["high", "very_high"]:
                    st.error(f"### ⚠️ {t['diseased_tag']} — {t['severity']}: {sev_label}")
                else:
                    st.warning(f"### ⚠️ {t['diseased_tag']} — {t['severity']}: {sev_label}")

            if result.is_uncertain:
                st.warning(t["uncertain_warning"])

            # Metric Cards
            m1, m2, m3 = st.columns(3)
            m1.metric(t["plant"], info["plant"])
            m2.metric(t["disease"], info["disease"])
            m3.metric(t["confidence"], f"{best.probability * 100:.1f}%")

            # Confidence Progress Bar
            st.progress(min(max(best.probability, 0.0), 1.0))

            # Top Predictions Breakdown
            with st.expander(t["top_preds"], expanded=False):
                for p in result.top_predictions:
                    st.write(f"• **{p.info['plant']} - {p.info['disease']}:** {p.probability * 100:.2f}%")
                    st.progress(min(max(p.probability, 0.0), 1.0))

            # Detailed Agronomic Insights
            st.markdown(f"#### 🌾 الإرشادات والتشخيص الزراعي التفصيلي" if lang == "ar" else "#### 🌾 Detailed Agronomic Insights")

            with st.expander(t["about"], expanded=True):
                st.markdown(f"**{info.get('description', '')}**")
                if not result.is_healthy and info.get("cause"):
                    st.markdown(f"• **{t['cause']}:** {info['cause']}")

            if info.get("symptoms"):
                with st.expander(t["symptoms"], expanded=False):
                    for sym in info["symptoms"]:
                        st.markdown(f"- {sym}")

            if not result.is_healthy and info.get("treatment"):
                with st.expander(t["treatment"], expanded=True):
                    for tr in info["treatment"]:
                        st.markdown(f"✅ {tr}")

            if not result.is_healthy and info.get("prevention"):
                with st.expander(t["prevention"], expanded=False):
                    for pr in info["prevention"]:
                        st.markdown(f"🛡️ {pr}")
        else:
            st.info("👈 اضغط على زر **(🔍 بدء الفحص والتشخيص)** لتحليل حالة الورقة وعرض التقرير الزراعي الكامل." if lang == "ar" else "👈 Click **(🔍 Analyze & Diagnose)** to run the AI diagnosis.")
