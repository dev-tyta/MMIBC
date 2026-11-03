"""Dataset sources.

Entries marked ``verified=False`` come from places I could not check from the build sandbox (no access to
Kaggle). ``mmibc download`` validates what it fetched against ``expected`` and reports precisely what is missing,
rather than failing later in a confusing way.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    kind: str  # "kaggle" | "url"
    ref: str  # kaggle slug or URL
    expected: tuple[str, ...]  # glob patterns (relative to the dataset dir); "a|b" means any-of
    note: str = ""
    verified: bool = False
    fallback_urls: tuple[str, ...] = ()


IMG = "**/*.png|**/*.jpg|**/*.jpeg"

REGISTRY: dict[str, DatasetSpec] = {
    "vindr_images": DatasetSpec(
        "vindr_images", "kaggle", "shantanughosh/vindr-mammogram-dataset-dicom-to-png", ("**/*.png",),
        note="VinDr-Mammo, DICOM already converted to PNG. Slug taken from the legacy fetch_data.py.", verified=True,
    ),
    "vindr_annotations": DatasetSpec(
        "vindr_annotations", "kaggle", "truthisneverlinear/vindr-mammo-annotations",
        ("**/breast-level_annotations.csv", "**/finding_annotations.csv"),
        note="VinDr-Mammo breast-level and finding-level annotation CSVs. Slug from the legacy fetch_data.py.", verified=True,
    ),
    "busi": DatasetSpec(
        "busi", "kaggle", "aryashah2k/breast-ultrasound-images-dataset", ("**/benign/*.png", "**/malignant/*.png"),
        note="BUSI. UNVERIFIED Kaggle mirror slug; the legacy code used the Cairo University URL (fallback).",
        fallback_urls=("https://scholar.cu.edu.eg/Dataset_BUSI.zip",),
    ),
    "bus_bra": DatasetSpec(
        "bus_bra", "url", "https://zenodo.org/records/8231412/files/BUSBRA.zip?download=1", (IMG,),
        note=(
            "BUS-BRA: 1,064 patients, biopsy-proven tumours, BI-RADS 2-5, segmentation masks, CC BY 4.0, ~134 MB zip on Zenodo "
            "(DOI 10.5281/zenodo.8231412). The exact file URL and the internal layout are unverified: after downloading run "
            "`mmibc inspect bus_bra`, then write raw/bus_bra/source_manifest.csv (see ultrasound.load_source_manifest)."
        ),
    ),
    "breast_lesions_usg": DatasetSpec(
        "breast_lesions_usg", "manual", "https://www.cancerimagingarchive.net/", (IMG,),
        note=(
            "BrEaST-Lesions-USG (Pawlowska et al., Sci Data 2024), hosted on TCIA. Intended as the EXTERNAL ultrasound test source. "
            "I could not open the dataset paper from my sandbox, so size and layout are unverified. Download from TCIA into "
            "raw/breast_lesions_usg/ and write source_manifest.csv."
        ),
    ),
    "cmmd": DatasetSpec(
        "cmmd", "manual", "https://www.cancerimagingarchive.net/", ("**/*.dcm|**/*.png",),
        note=(
            "CMMD (Chinese Mammography Database, TCIA): biopsy-confirmed benign/malignant mammograms with CC/MLO views - a "
            "pathology-labelled alternative to VinDr's BI-RADS labels. Details unverified from my sandbox (page blocked). DICOM "
            "download via the TCIA NBIA Data Retriever; convert to PNG before use."
        ),
    ),
    "bus_uclm": DatasetSpec(
        "bus_uclm", "manual", "https://www.nature.com/articles/s41597-025-04562-3", (IMG,),
        note="BUS-UCLM (Scientific Data 2025): ultrasound lesion-segmentation dataset. Could not be opened from my sandbox; verify size, labels and licence first.",
    ),
    "kau_bcmd": DatasetSpec(
        "kau_bcmd", "kaggle", "orvile/kau-bcmd-mamography-dataset", (IMG,),
        note=(
            "KAU-BCMD. The paper reports 1,416 mammogram cases (4 views) and 205 ultrasound cases that correspond to part "
            "of the mammogram cases. This mirror's layout is not known to me: run `mmibc inspect kau_bcmd` to see "
            "whether the ultrasound subset and the case IDs linking it to the mammograms are present."
        ),
        verified=True,  # slug from legacy code; *contents* still need inspection
    ),
}
