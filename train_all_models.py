# train_all_models.py
# Kevin P. Klier – OsteoID.ai final training pipeline (Nov 2025)
# Run once → creates all 18 pickle files + prints real holdout accuracies

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import procrustes
from imblearn.over_sampling import SMOTE
import pickle
from pathlib import Path

# ------------------------------------------------------------------
# Bulletproof parser (your original – perfect)
# ------------------------------------------------------------------
def load_morphofile(filepath):
    names, landmarks = [], []
    current_name = None
    current_lms = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(('#', "'#")):
            if current_name and current_lms:
                landmarks.append(np.array(current_lms))
                names.append(current_name)
            current_name = stripped.strip("#' ")
            current_lms = []
            continue
        try:
            coords = [float(x) for x in stripped.split()]
            if len(coords) == 3:
                current_lms.append(coords)
        except:
            pass
    if current_name and current_lms:
        landmarks.append(np.array(current_lms))
        names.append(current_name)
    return names, np.stack(landmarks)

# ------------------------------------------------------------------
# Name parser (one version works for all three bones)
# ------------------------------------------------------------------
def parse_name(name, bone_keyword):
    parts = name.split('_')
    bone_idx = parts.index(bone_keyword)
    sex = parts[bone_idx - 1]           # M or F
    side = parts[-1][-1]                # L or R
    species = '_'.join(parts[1:bone_idx-1])
    return species, sex, side

# ------------------------------------------------------------------
# Bone configuration
# ------------------------------------------------------------------
bones = {
    "clavicle": {"file": "MorphoFileClavicle_ANHP.txt", "keyword": "clavicle"},
    "scapula":  {"file": "MorphoFileScapula_ANHP.txt",  "keyword": "scapula"},
    "humerus":  {"file": "MorphoFileHumerus_ANHP.txt",  "keyword": "humerus"}
}

# ------------------------------------------------------------------
# Train everything
# ------------------------------------------------------------------
for bone, info in bones.items():
    print(f"\nTraining {bone.upper()} (185 specimens)")

    names, landmarks = load_morphofile(info["file"])
    species_list, sex_list, side_list = [], [], []

    for n in names:
        sp, sex, side = parse_name(n, info["keyword"])
        species_list.append(sp)
        sex_list.append(sex)
        side_list.append(side)

    # GPA
    mean_shape = landmarks.mean(axis=0)
    aligned = np.zeros_like(landmarks)
    for i in range(len(landmarks)):
        _, aligned[i], _ = procrustes(mean_shape, landmarks[i])

    flat = aligned.reshape(len(aligned), -1)
    pca = PCA(n_components=10, random_state=42)
    features = pca.fit_transform(flat)

    # Train/test split (same as your README)
    X_tr, X_te, y_sp_tr, y_sp_te, y_sex_tr, y_sex_te, y_side_tr, y_side_te = train_test_split(
        features, species_list, sex_list, side_list,
        test_size=0.2, random_state=42, stratify=species_list
    )

    # Species model with SMOTE
    le = LabelEncoder()
    y_sp_tr_enc = le.fit_transform(y_sp_tr)
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_tr, y_sp_tr_enc)
    model_species = RandomForestClassifier(
        n_estimators=1000, class_weight='balanced', random_state=42, n_jobs=-1
    )
    model_species.fit(X_res, y_res)
    sp_acc = accuracy_score(le.transform(y_sp_te), model_species.predict(X_te))

    # Sex & Side (no SMOTE)
    model_sex = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    model_sex.fit(X_tr, y_sex_tr)
    sex_acc = accuracy_score(y_sex_te, model_sex.predict(X_te))

    model_side = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    model_side.fit(X_tr, y_side_tr)
    side_acc = accuracy_score(y_side_te, model_side.predict(X_te))

    print(f"  → Holdout: Species {sp_acc:.1%} | Sex {sex_acc:.1%} | Side {side_acc:.1%}")

    # Save everything in correct folder
    out_dir = Path("models") / bone
    out_dir.mkdir(parents=True, exist_ok=True)

    pickle.dump(mean_shape, open(out_dir / f"mean_shape_{bone}.pkl", "wb"))
    pickle.dump(pca, open(out_dir / f"pca_{bone}.pkl", "wb"))
    pickle.dump(model_species, open(out_dir / f"model_species_{bone}.pkl", "wb"))
    pickle.dump(model_sex, open(out_dir / f"model_sex_{bone}.pkl", "wb"))
    pickle.dump(model_side, open(out_dir / f"model_side_{bone}.pkl", "wb"))
    pickle.dump(le, open(out_dir / f"le_species_{bone}.pkl", "wb"))

    print(f"  → Saved to {out_dir}/\n")

print("All done! OsteoID.ai is fully trained and ready for deployment.")