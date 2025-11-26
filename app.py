import streamlit as st
import trimesh
import numpy as np
import pickle
import plotly.graph_objects as go
from scipy.spatial import procrustes, KDTree
import onnxruntime as ort
import os

@st.cache_resource
def load_model_components(bone: str):
    model_dir = f"models_onnx/{bone}"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Missing {model_dir}")
    
    mean_shape = pickle.load(open(f"{model_dir}/mean_shape_{bone}.pkl", "rb"))
    pca = pickle.load(open(f"{model_dir}/pca_{bone}.pkl", "rb"))
    le_species = pickle.load(open(f"{model_dir}/le_species_{bone}.pkl", "rb"))
    
    sess_species = ort.InferenceSession(f"{model_dir}/model_species_{bone}.onnx")
    sess_sex = ort.InferenceSession(f"{model_dir}/model_sex_{bone}.onnx")
    sess_side = ort.InferenceSession(f"{model_dir}/model_side_{bone}.onnx")
    
    return mean_shape, pca, le_species, sess_species, sess_sex, sess_side

st.set_page_config(page_title="OsteoID.ai", layout="centered")
st.title("OsteoID.ai")
st.markdown("**Primate Shoulder Bone Classifier — Beta**  \nKevin P. Klier")

bone_choice = st.selectbox("Bone type", ["Auto-detect", "clavicle", "scapula", "humerus"])
uploaded_file = st.file_uploader("Upload raw .ply file", type="ply")

if uploaded_file is not None:
    try:
        mesh = trimesh.load(trimesh.util.wrap_as_stream(uploaded_file.getvalue()), file_type='ply')
        verts = np.asarray(mesh.vertices).astype(np.float64)

        if len(verts) < 100:
            st.error("Invalid or empty mesh.")
            st.stop()

        centroid = verts.mean(axis=0)
        verts -= centroid

        if bone_choice == "Auto-detect":
            candidates = {}
            for b in ["clavicle", "scapula", "humerus"]:
                try:
                    mean_shape, pca, le, sess_sp, _, _ = load_model_components(b)
                    sample = verts[np.random.choice(len(verts), size=min(10000, len(verts)), replace=False)]
                    mesh_cs = np.sqrt(np.sum(verts**2) / len(verts))
                    tmpl_cs = np.sqrt(np.sum(mean_shape**2) / len(mean_shape))
                    scale_factor = tmpl_cs / mesh_cs if mesh_cs > 0 else 1.0
                    scaled = sample * scale_factor
                    flat = scaled[:mean_shape.shape[0]].flatten().reshape(1, -1)
                    pca_feats = pca.transform(flat).astype(np.float32)
                    cs_norm = (mesh_cs - pca.mean_[0]) / pca.scale_[0] if hasattr(pca, 'scale_') else 0
                    feats = np.hstack([pca_feats, [[cs_norm]]]).astype(np.float32)
                    pred = sess_sp.run(None, {"float_input": feats})[0]
                    conf = np.max(pred)
                    candidates[b] = conf
                except Exception as e:
                    st.warning(f"Auto-detect failed for {b}: {e}")
                    continue
            if not candidates:
                st.error("Auto-detect failed — using manual selection")
                bone = bone_choice.lower()
            else:
                bone = max(candidates, key=candidates.get)
                st.info(f"Auto-detected: **{bone.capitalize()}**")
        else:
            bone = bone_choice

        mean_shape, pca, le_species, sess_species, sess_sex, sess_side = load_model_components(bone)

        st.write(f"**Processing as {bone.capitalize()}** — {len(verts):,} vertices")

        mesh_cs = np.sqrt(np.sum(verts**2) / len(verts))
        tmpl_cs = np.sqrt(np.sum(mean_shape**2) / len(mean_shape))
        verts = verts * (tmpl_cs / mesh_cs if mesh_cs > 0 else 1.0)

        def align_template(template, points, max_iter=40):
            src = template.copy()
            tree = KDTree(points)
            for _ in range(max_iter):
                _, idx = tree.query(src)
                closest = points[idx]
                _, src, _ = procrustes(closest, src)
            return src

        sample_pts = verts[np.random.choice(len(verts), size=min(20000, len(verts)), replace=False)]
        aligned_lms = align_template(mean_shape, sample_pts)
        _, aligned_lms, _ = procrustes(mean_shape, aligned_lms)

        pca_feats = pca.transform(aligned_lms.flatten().reshape(1, -1)).astype(np.float32)
        cs_norm = (mesh_cs - pca.mean_[0]) / pca.scale_[0] if hasattr(pca, 'scale_') else 0
        feats = np.hstack([pca_feats, [[cs_norm]]]).astype(np.float32)

        prob_species = sess_species.run(None, {"float_input": feats})[1][0]
        pred_species_idx = int(np.argmax(prob_species))
        pred_species_label = le_species.inverse_transform([pred_species_idx])[0]
        conf_species = prob_species[pred_species_idx] * 100

        prob_sex = sess_sex.run(None, {"float_input": feats})[1][0]
        pred_sex_label = "Male" if np.argmax(prob_sex) == 1 else "Female"
        conf_sex = np.max(prob_sex) * 100

        prob_side = sess_side.run(None, {"float_input": feats})[1][0]
        pred_side_label = "Left" if np.argmax(prob_side) == 0 else "Right"
        conf_side = np.max(prob_side) * 100

        if min(conf_species, conf_sex, conf_side) < 65:
            st.warning("Low confidence — alignment may be poor. Try re-centering in MeshLab/Blender.")

        st.success(f"**Bone**: {bone.capitalize()}")
        st.success(f"**Species**: {pred_species_label} ({conf_species:.1f}% confidence)")

        st.write("**Top 3 species**")
        top3 = np.argsort(prob_species)[-3:][::-1]
        for i in top3:
            sp = le_species.inverse_transform([i])[0]
            st.write(f"• {sp} — {prob_species[i]*100:.1f}%")

        st.success(f"**Sex**: {pred_sex_label} ({conf_sex:.1f}% confidence)")
        st.success(f"**Side**: {pred_side_label} ({conf_side:.1f}% confidence)")

        fig = go.Figure(data=[
            go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2], color='lightgray', opacity=0.4, name='Bone'),
            go.Scatter3d(x=aligned_lms[:,0], y=aligned_lms[:,1], z=aligned_lms[:,2],
                         mode='markers', marker=dict(size=6, color='red'), name='Aligned landmarks')
        ])
        fig.update_layout(scene_aspectmode='data', height=700)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

else:
    st.info("Upload a raw .ply shoulder bone (clavicle, scapula, or humerus) to begin.")
    st.markdown("*Non-human primates only · No landmarking required*")

st.markdown("---")
st.caption("Based on M.A. research at University at Buffalo under advisement of Dr. Noreen von Cramon-Taubadel")
