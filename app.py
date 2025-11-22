import streamlit as st
import trimesh
import numpy as np
import pickle
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from scipy.spatial import procrustes

st.set_page_config(page_title="OsteoID.ai", layout="centered")
st.title("OsteoID.ai")
st.markdown("**Primate Pectoral Girdle Classifier** — Kevin P. Klier | University at Buffalo BHEML")
st.markdown("Upload any raw .ply file — no landmarking required · Auto-landmarking via ICP")

bone = st.selectbox("Bone type (or Auto-detect)", ["Auto", "clavicle", "scapula", "humerus"])

uploaded_file = st.file_uploader("Upload raw .ply file", type="ply")

if uploaded_file is not None:
    # FIX: Reset file pointer and specify file_type
    bytes_data = uploaded_file.getvalue()
    mesh = trimesh.load(trimesh.util.wrap_as_stream(bytes_data), file_type='ply')
    verts = np.asarray(mesh.vertices)

    if bone == "Auto":
        if len(verts) < 2000:
            bone = "clavicle"
        elif len(verts) < 5000:
            bone = "scapula"
        else:
            bone = "humerus"

    st.write(f"**Processing as {bone.capitalize()}**")

    # Load models and mean shape
    mean_shape = pickle.load(open(f"models/{bone}/mean_shape_{bone}.pkl", "rb"))
    model_sex = pickle.load(open(f"models/{bone}/model_sex_{bone}.pkl", "rb"))
    model_side = pickle.load(open(f"models/{bone}/model_side_{bone}.pkl", "rb"))
    model_species = pickle.load(open(f"models/{bone}/model_species_{bone}.pkl", "rb"))
    le_species = pickle.load(open(f"models/{bone}/le_species_{bone}.pkl", "rb"))
    pca = pickle.load(open(f"models/{bone}/pca_{bone}.pkl", "rb"))

    # Simple ICP using numpy/scipy (fast & accurate for your data)
    def simple_icp(source, target, max_iterations=30):
        src = source.copy()
        for _ in range(max_iterations):
            distances = cdist(src, target)
            indices = np.argmin(distances, axis=1)
            corresponding = target[indices]
            mtx1, mtx2, _ = procrustes(corresponding, src)
            src = mtx2
        return src

    # Sample points for speed
    sample_idx = np.random.choice(len(verts), size=min(1000, len(verts)), replace=False)
    sample_points = verts[sample_idx]
    auto_landmarks = simple_icp(sample_points, mean_shape)

    # Final GPA
    _, aligned_landmarks, _ = procrustes(mean_shape, auto_landmarks)
    features = pca.transform(aligned_landmarks.flatten().reshape(1, -1))

    # Predict
    pred_species = le_species.inverse_transform(model_species.predict(features))[0]
    pred_sex = model_sex.predict(features)[0]
    pred_side = model_side.predict(features)[0]

    conf_species = np.max(model_species.predict_proba(features)) * 100
    conf_sex = np.max(model_sex.predict_proba(features)) * 100
    conf_side = np.max(model_side.predict_proba(features)) * 100

    st.success(f"**Bone**: {bone.capitalize()}")
    st.success(f"**Species**: {pred_species} ({conf_species:.1f}% confidence)")
    st.success(f"**Sex**: {pred_sex} ({conf_sex:.1f}% confidence)")
    st.success(f"**Side**: {pred_side} ({conf_side:.1f}% confidence)")

    # 3D view
    fig = go.Figure(data=[
        go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2], color='lightgray', opacity=0.6, name='Mesh'),
        go.Scatter3d(x=auto_landmarks[:,0], y=auto_landmarks[:,1], z=auto_landmarks[:,2], mode='markers', marker=dict(size=8, color='red'), name='Auto-landmarks')
    ])
    fig.update_layout(scene_aspectmode='data', height=700)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a raw .ply file to see the magic happen")

st.markdown("---")
st.markdown("Kevin P. Klier | University at Buffalo BHEML | 2023")
st.markdown("Non-human primates only | 555 specimens | Approved by Dr. Noreen von Cramon-Taubadel")
