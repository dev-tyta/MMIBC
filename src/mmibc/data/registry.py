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
    extra_files: tuple[tuple[str, str], ...] = ()  # (file name, url): plain files saved next to the archive (e.g. a label spreadsheet)


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
        note=(
            "BUSI. UNVERIFIED Kaggle mirror slug. The Cairo University URL from the legacy code no longer serves the zip "
            "(Nov 2026: it returned a ~19 kB web page), so it only remains as a last-resort fallback. If the Kaggle slug fails, "
            "download BUSI by hand into <local>/raw/busi/ (it must contain benign/, malignant/ and normal/ folders)."
        ),
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
        "breast_lesions_usg", "url",
        "https://www.cancerimagingarchive.net/wp-content/uploads/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip",
        (IMG,),
        note=(
            "BrEaST-Lesions-USG (Pawlowska et al., Sci Data 2024), hosted on TCIA. Intended as the EXTERNAL ultrasound test source. "
            "Two files are fetched: the images-and-masks zip and the clinical-data spreadsheet (the zip name suggests it holds images "
            "and masks only, so the labels are expected in the spreadsheet). Both URLs come from the maintainer; neither could be opened "
            "from my sandbox, so the zip layout, the spreadsheet columns and the licence are UNVERIFIED. Run `mmibc inspect "
            "breast_lesions_usg` (it prints the spreadsheet's columns and value counts), then write raw/breast_lesions_usg/source_manifest.csv "
            "(uid, raw_path, class_name[, mask_paths, patient_id])."
        ),
        extra_files=(
            (
                "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx",
                "https://www.cancerimagingarchive.net/wp-content/uploads/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx",
            ),
        ),
    ),
    "cmmd": DatasetSpec(
        "cmmd", "kaggle", "tommyngx/cmmd2022", (IMG + "|**/*.dcm",),
        note=(
            "CMMD (Chinese Mammography Database): MAMMOGRAPHY, not ultrasound. Biopsy-confirmed benign/malignant with CC/MLO "
            "views, a pathology-labelled alternative to VinDr's BI-RADS labels. The Kaggle page title was confirmed from the "
            "link you supplied; the file format, folder layout, label file and licence are UNKNOWN (page body unreadable from "
            "my sandbox), so run `mmibc inspect cmmd` after downloading. The original is on TCIA (NBIA Data Retriever) if the "
            "mirror is incomplete. Downloadable and inspectable, but no `prepare` step consumes it yet."
        ),
    ),
    "bus_uclm": DatasetSpec(
        "bus_uclm", "kaggle", "orvile/bus-uclm-breast-ultrasound-dataset", (IMG,),
        note=(
            "BUS-UCLM (Scientific Data 2025), breast ultrasound lesion segmentation. The Kaggle page title was confirmed from the "
            "link you supplied; file layout, how classes are encoded (class folders vs. colour-coded masks), patient IDs and "
            "licence are UNKNOWN (page body unreadable from my sandbox). Run `mmibc inspect bus_uclm`; if it is not "
            "class folders, write raw/bus_uclm/source_manifest.csv (uid, raw_path, class_name[, mask_paths, patient_id]) before use. "
            "Without patient IDs the audit cannot rule out patient-level leakage within this source."
        ),
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
