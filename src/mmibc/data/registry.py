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
    md5: str = ""  # published MD5 of the main zip: checked after downloading, and for a zip placed in the folder by hand


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
        "busi", "kaggle", "sabahesaraki/breast-ultrasound-images-dataset", ("**/benign/*.png", "**/malignant/*.png"),
        note=(
            "BUSI (Al-Dhabyani et al.). Kaggle mirror chosen by the maintainer; the page exists but its file listing could not be "
            "read from my sandbox, so the layout is UNVERIFIED: expect benign/, malignant/ and normal/ folders with a *_mask.png next to "
            "each image (masks are skipped automatically). The original Cairo University link no longer serves the file, so there "
            "is no URL fallback. If the download fails, get BUSI by hand and put the zip (or the folders) into this dataset's raw folder."
        ),
    ),
    "bus_bra": DatasetSpec(
        "bus_bra", "url", "https://zenodo.org/records/8231412/files/BUSBRA.zip?download=1", (IMG,),
        note=(
            "BUS-BRA: 1,064 patients, biopsy-proven tumours, BI-RADS 2-5, segmentation masks, CC BY 4.0 (DOI 10.5281/zenodo.8231412). "
            "The record page lists one open file, BUSBRA.zip (133.9 MB, MD5 1f8b2be6476d58fc97bfb5e5a1ea9bab), at the URL used here. "
            "If Zenodo refuses the request from your runtime (HTTP 403), download BUSBRA.zip in a browser and put it into this "
            "dataset's raw folder on Drive: `mmibc download bus_bra` unpacks and checks it. Layout (confirmed on a real run): "
            "BUSBRA/{Images,Masks,bus_data.csv}; `prepare us` builds raw/bus_bra/source_manifest.csv from bus_data.csv by itself "
            "(`mmibc manifest bus_bra` does it explicitly), with Case as patient_id."
        ),
        fallback_urls=(
            "https://zenodo.org/api/records/8231412/files/BUSBRA.zip/content",  # the documented REST download endpoint
            "https://zenodo.org/record/8231412/files/BUSBRA.zip?download=1",  # older URL form
        ),
        md5="1f8b2be6476d58fc97bfb5e5a1ea9bab",
    ),
    "breast_lesions_usg": DatasetSpec(
        "breast_lesions_usg", "url",
        "https://www.cancerimagingarchive.net/wp-content/uploads/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip",
        (IMG,),
        note=(
            "BrEaST-Lesions-USG (Pawlowska et al., Sci Data 2024), hosted on TCIA. Intended as the EXTERNAL ultrasound test source. "
            "Two files are fetched: the images-and-masks zip and the clinical-data spreadsheet (the zip name suggests it holds images "
            "and masks only, so the labels are expected in the spreadsheet). Both URLs come from the maintainer and the download was confirmed to work from Colab (69.9 MB zip + 40 kB "
            "spreadsheet). Layout: a flat folder of caseNNN.png / caseNNN_tumor.png and a spreadsheet with CaseID, Image_filename, "
            "Mask_tumor_filename, Classification (benign 154 / malignant 98 / normal 4, one image per case; 55 benign-looking cases are "
            "confirmed by follow-up rather than biopsy). `prepare us` builds raw/breast_lesions_usg/source_manifest.csv from it "
            "(`mmibc manifest breast_lesions_usg` does it explicitly). The licence is still UNVERIFIED."
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
            "views, a pathology-labelled alternative to VinDr's BI-RADS labels. `mmibc prepare cmmd` / `data.mammo_source=cmmd` use it. "
            "It needs the TCIA layout (confirmed on a real copy): the CMMD/ folder of .dcm files, CMMD_clinicaldata_revision.xlsx "
            "(the labels are NOT in the DICOM files) and optionally TCIA's metadata.csv, anywhere under raw/cmmd/. The Kaggle mirror "
            "named here is UNVERIFIED: its format (DICOM or converted images), folder layout and label file are unknown, and converted "
            "images lose the breast side and view, which `prepare cmmd` refuses. If the mirror lacks the DICOMs or the spreadsheet, "
            "download CMMD from TCIA (NBIA Data Retriever) and place it in raw/cmmd/."
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
