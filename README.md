# finance-triple-extraction-Full-Project-code-README-

Full end-to-end project to extract triples (h, r, t) from financial narrative text using spaCy (rule-based + basic NER) and optionally a transformer-based relation classifier. Includes code, sample data, and a demo notebook.

finance-triple-extraction/
├── README.md
├── requirements.txt
├── data/
│ ├── sample_texts.txt
│ └── annotations/ # optional: for future finetuning
├── src/
│ ├── preprocess.py
│ ├── relation_rules.py
│ ├── relation_model.py
│ ├── triples.py
│ └── utils.py
├── notebooks/
│ └── demo.ipynb
└── outputs/
├── triples.json
└── triples.csv
