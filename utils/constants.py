
DEFECT_CLASSES = {
    0: 'pleat',
    1: 'stain',
    2: "weaving",
    3: "hard_pleat",
    4: "ignore"
}
MAPPING_CLASSES = {
    "Không có lỗi": "Khong_co_loi",
    "Anh mau": "Anh_mau",
    "Loi soi": "Loi_soi",
    "Nep gap vai": "Nep_gap_vai",
    "Cham do": "Cham_do"
}

# Canonicalizes every label vocabulary seen in raw device JSON ("gt" worker
# labels, "pos" on-device production predictions -- old VN-with-space,
# old VN-with-underscore, and new English DEFECT_CLASSES names) onto the
# new model's English class names, so ground truth / production / new_model
# can be IoU-matched by class without silently zeroing out precision/recall
# on string-format mismatches (e.g. "Cham do" vs "Cham_do" vs "stain").
# "Anh_mau" / "Dau_keo" have no new-model equivalent (out of scope for the
# June retrain) and are left as their own canonical classes.
CANONICAL_LABELS = {
    "Cham do": "stain",
    "Cham_do": "stain",
    "Loi soi": "weaving",
    "Loi_soi": "weaving",
    "Nep gap vai": "pleat",
    "Nep_gap_vai": "pleat",
    "Anh mau": "Anh_mau",
    "Dau keo": "Dau_keo",
    "Không có lỗi": "Khong_co_loi",
}