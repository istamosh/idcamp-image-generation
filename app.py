import streamlit as st
import logic

st.set_page_config(page_title="Image Generator", layout="centered", page_icon="🖼️")
st.title("Image Generation with Stable Diffusion")

@st.cache_resource
def get_models():
    return logic.load_models_cached()

try:
    pipe_txt2img, _ = get_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

prompt = st.text_input(
    "Prompt",
    value="a futuristic city at sunset, ultra detailed, cinematic lighting"
)
negative_prompt = st.text_input(
    "Negative Prompt",
    value="blurry, low quality, distorted, bad anatomy"
)

guidance_scale = st.slider("guidance_scale", min_value=1.0, max_value=20.0, value=7.5, step=0.5)
num_inference_steps = st.slider("num_inference_steps", min_value=10, max_value=60, value=30, step=1)

if st.button("Generate", type="primary"):
    if not prompt.strip():
        st.warning("Prompt tidak boleh kosong.")
    else:
        with st.spinner("Generating image..."):
            logic.flush_memory()
            generated_images = logic.generate_image(
                pipe=pipe_txt2img,
                prompt=prompt,
                neg_prompt=negative_prompt,
                seed=42,
                steps=num_inference_steps,
                cfg=guidance_scale,
                num_images=1,
                scheduler_name="Euler A"
            )

        if generated_images:
            st.image(generated_images[0], caption="Generated Image", use_container_width=True)
        else:
            st.error("Gagal menghasilkan gambar.")
