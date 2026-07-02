ethiopian-scales/
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/                      ← the 5 phase notebooks (run, with outputs)
│   ├── 01_preprocessing.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_clustering.ipynb
│   ├── 04_classification.ipynb
│   └── 05_explainability.ipynb
│
├── src/                            ← the .py scripts, grouped by phase
│   ├── preprocessing.py
│   ├── loader.py
│   ├── inspect_data.py
│   ├── features_logmel.py
│   ├── features_mfcc.py
│   ├── features_chroma.py
│   ├── chroma_highlight.py
│   ├── build_features.py
│   ├── cluster_kmeans.py
│   ├── cluster_chroma_only.py
│   ├── visualize_pca.py
│   ├── classify_svm.py
│   ├── classify_rf.py
│   ├── confusion_matrices.py
│   ├── explain_feature_importance.py
│   ├── explain_shap.py
│   └── explain_lime.py
│
├── figures/                        ← generated plots (committed, for the report)
│
├── features/                       ← X.npy, y.npy, features.csv (see note)
│
├── Kaleabe Seifu APR Project Report.pdf           ← the written deliverables
│ 
│   
│
└── data/                           ← NOT committed (see .gitignore)
    └── raw/
        ├── tizita/
        └── bati/
