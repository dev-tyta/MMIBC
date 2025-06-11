

## Multimodal Breast Cancer Imaging Dataset - MMIBC

This dataset combines publicly available mammography and ultrasound imaging data for the development and evaluation of multimodal deep learning models for breast cancer diagnosis, with a focus on interpretability and applicability in resource-limited settings.

## Dataset Sources

This dataset is compiled from the following publicly available sources:

* **VinDr-Mammo:** A large-scale benchmark dataset for computer-aided detection and diagnosis in full-field digital mammography.
    * **Source:** [https://doi.org/10.13026/br2v-7517](https://doi.org/10.13026/br2v-7517)
    * **Reference:** Pham, H. H., Nguyen-Trung, H., & Nguyen, H. Q. (2022). VinDr-Mammo: A large-scale benchmark dataset for computer-aided detection and diagnosis in full-field digital mammography (version 1.0.0). *PhysioNet*.

* **BUSI (Breast Ultrasound Images) Dataset:** A well-curated collection of breast ultrasound images.
    * **Source:** [https://doi.org/10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863)
    * **Reference:** Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). Dataset of breast ultrasound images. *Data in Brief*, 28, 104863.

* **KAU-BCMD (King Abdulaziz University Breast Cancer Mammogram Dataset):** Includes paired mammography and ultrasound images for a subset of cases, valuable for multimodal evaluation.
    * **Source:** [https://doi.org/10.3390/data6110111](https://doi.3390/data6110111)
    * **Reference:** Alsolami, A. S., Shalash, W., Alsaggaf, W., Ashoor, S., Refaat, H., & Elmogy, M. (2021). King Abdulaziz University Breast Cancer Mammogram Dataset (KAU-BCMD). *Data*, 6(11), 111.

## Dataset Structure

The dataset is organized into folders based on modality and data split. The proposed structure is as follows:
```
~/mmibc/
├── mammo/
│   ├── train/
│   │   ├── benign/
│   │   └── malignant/
│   ├── validation/
│   │   ├── benign/
│   │   └── malignant/
│   └── test/
│       ├── benign/
│       └── malignant/
├── ultrasound/
│   ├──images/
│   │   |── train/
│   │   │       ├── benign/
│   │   │       └── malignant/
│   │   ├── validation/
│   │   │       ├── benign/
│   │   │       └── malignant/
│   │   └── test/
│   │           ├── benign/
│   │           └── malignant/
│   ├──masks/
│   │   |── train/
│   │   │       ├── benign/
│   │   │       └── malignant/
│   │   ├── validation/
│   │   │       ├── benign/
│   │   │       └── malignant/
│   │   └── test/
│   │           ├── benign/
│   │           └── malignant/
│
└── kau_bcmd/
│    ├── train/        (Paired mammo and US images, potentially linked by ID)
│    ├── validation/   (Paired mammo and US images)
│    └── test/         (Paired mammo and US images)
```

Within each split (`train`, `validation`, `test`), images are further categorized by diagnosis (`benign`, `malignant`). The KAU-BCMD dataset, being used for multimodal evaluation, will contain paired images. The structure for KAU-BCMD might need to include a mapping or consistent naming convention to link the mammogram and ultrasound images for the same patient/case.

## Dataset Contents

* **Images:** DICOM or common image formats (PNG, JPG) for mammography and ultrasound images.
* **Annotations:** Original annotations provided with the datasets, which may include bounding boxes, lesion types, and BI-RADS assessments.
* **Labels:** Binary labels indicating the diagnosis (benign or malignant).

## Usage

This dataset is intended for training and evaluating deep learning models for breast cancer diagnosis using multimodal imaging. The structured format facilitates easy loading and processing using standard deep learning libraries and Hugging Face's `datasets` library.

## License

Please refer to the original licenses of the VinDr-Mammo, BUSI, and KAU-BCMD datasets for specific terms of use.

## Citation

Please cite the original dataset sources when using this compiled dataset in your research.

```bibtex
@article{pham2022vindr,
  title={VinDr-Mammo: A large-scale benchmark dataset for computer-aided detection and diagnosis in full-field digital mammography},
  author={Pham, Hai H and Nguyen-Trung, Hieu and Nguyen, Hoang Q},
  journal={PhysioNet},
  year={2022}
}

@article{al2020dataset,
  title={Dataset of breast ultrasound images},
  author={Al-Dhabyani, Waleed and Gomaa, Mohamed and Khaled, Hossam and Fahmy, Amr},
  journal={Data in Brief},
  volume={28},
  pages={104863},
  year={2020},
  publisher={Elsevier}
}

@article{alsolami2021king,
  title={King Abdulaziz University Breast Cancer Mammogram Dataset (KAU-BCMD)},
  author={Alsolami, Abdulrahman S and Shalash, Walid and Alsaggaf, Walid and Ashoor, Sara and Refaat, Hesham and Elmogy, Mohammed},
  journal={Data},
  volume={6},
  number={11},
  pages={111},
  year={2021},
  publisher={MDPI}
}
```