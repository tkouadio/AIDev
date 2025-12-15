# 🤖 Prédiction de l’acceptation des Pull Requests générées ou assistées par l’IA  
**Projet final du cours l’ingénierie de mise en production des versions logicielles (MGL869) à la maîtrise en génie logiciel. – AIDev / Mining Software Repositories**

---

## 📌 Contexte du projet

Ce projet s’inscrit dans le cadre du projet **AIDev**, proposé à partir du **Mining Challenge MSR** :

- 🔗 AIDev: https://github.com/SAILResearch/AI_Teammates_in_SE3  
- 🔗 MSR Mining Challenge: https://2026.msrconf.org/track/msr-2026-mining-challenge  
- 🔗 Dataset Zenodo: https://zenodo.org/records/16919272  

L’objectif général du projet **AIDev** est d’analyser l’impact des agents d’Intelligence Artificielle (IA) sur la productivité et la qualité du développement logiciel, à partir de données réelles issues de GitHub.

---

## 🎯 Objectif du projet

L’objectif spécifique de ce projet est de répondre à la question suivante :

**Peut-on prédire si une Pull Request générée ou assistée par un agent IA sera acceptée ou rejetée ?**

Pour cela, j'ai construit :
- un pipeline complet de données,  
- un modèle de machine learning interprétable,  
- et une analyse explicative des facteurs influençant l’acceptation des Pull Requests.

---

## ❓ Questions de recherche (Research Questions)

- **RQ1 :** Quelles caractéristiques différencient les Pull Requests acceptées des Pull Requests rejetées ?  
- **RQ2 :** Est-il possible de prédire l’acceptation d’une Pull Request IA à partir de ses métriques ?  
- **RQ3 :** Les agents IA ont-ils un impact significatif sur l’acceptation des Pull Requests, comparativement aux facteurs humains et techniques ?

---

## 🧠 Approche et méthodologie

### 🔹 1. Chargement des données

Les données proviennent du dataset officiel **AIDev (Zenodo)**, comprenant :
- Pull Requests  
- Commits et détails de commits  
- Reviews et commentaires  
- Informations sur les auteurs  
- Informations sur les dépôts  
- Agents IA associés aux PR  

---

### 🔹 2. Feature Engineering

J'ai extrait plusieurs catégories de métriques :

**📄 Structure de la PR**
- `title_length`, `body_length`

**🔧 Taille et complexité du code**
- `commits`, `changed_files`
- `additions`, `deletions`, `total_changes`

**👥 Collaboration**
- `num_comments`
- `num_reviews`
- `num_review_comments`
- `num_reviewers_unique`

**⏱ Temporalité**
- `pr_duration_days`
- `created_hour`

**👤 Auteur**
- `followers`
- `public_repos`
- `author_tenure_days`

**🏗 Dépôt**
- `forks`
- `stars`

**🤖 Agents IA**
Encodage one-hot :  
- `agent_OpenAI_Codex`, `agent_Copilot`, `agent_Devin`, `agent_Cursor`, `agent_Claude_Code`

---

### 🔹 3. Modèle de Machine Learning

J'ai utilisé un **RandomForestClassifier** pour les raisons suivantes :
- robuste face aux données hétérogènes,  
- capable de capturer des relations non linéaires,  
- compatible avec **SHAP** pour l’interprétabilité.

**📊 Split des données :**
- 80 % entraînement  
- 20 % test  

---

## 📈 Résultats

### 🎯 Performance du modèle

- **Accuracy globale :** ~ 88 %  
- **F1-score PR acceptées :** ~ 0.92  
- **F1-score PR rejetées :** ~ 0.77  

➡️ Le modèle prédit très bien les PR acceptées, les PR rejetées étant plus difficiles car minoritaires.

---

### 🔍 Interprétation avec SHAP

L’analyse SHAP montre que :

✅ Les facteurs les plus influents sont :
- durée de vie de la PR,  
- taille du patch,  
- nombre de reviewers uniques,  
- expérience de l’auteur.  

✅ Les agents IA ont un effet réel mais marginal :
- **OpenAI Codex** a un léger effet positif,  
- **Copilot** est globalement neutre,  
- **Devin**, **Cursor** et **Claude Code** ont une influence très faible.  

➡️ Les agents IA ne sont pas les facteurs déterminants de l’acceptation.

---

## ⚠️ Limites du projet

- ❌ Absence d’information sur la présence de tests.  
- ❌ Absence de métriques sur la qualité sémantique du code.  
- ❌ Agents IA auto-déclarés → bruit possible.  
- ❌ Le modèle ne capture pas l’intention du mainteneur ni le contexte du projet.

---

## 📂 Structure du projet

```

AIDev/
├── data/                     # Données brutes
├── scripts/
│   ├── load_data.py
│   ├── feature_engineering.py
│   ├── merge_all.py
│   ├── train_model.py
│   └── evaluate_model.py
├── notebooks/
│   └── AIDev_Pipeline.ipynb
├── artifacts/
│   ├── model_rf.joblib
│   ├── model_features.csv
│   ├── shap_summary_bar.png
│   ├── permutation_importances.csv
│   └── classification_report.txt
├── requirements.txt
├── .gitignore
└── README.md

````

---

## ▶️ Comment exécuter le projet

### 1️⃣ Installer les dépendances
```bash
pip install -r requirements.txt
````

### 2️⃣ Entraîner le modèle

```bash
python scripts/train_model.py
```

### 3️⃣ Évaluer le modèle et générer les graphiques

```bash
python scripts/evaluate_model.py
```

---

## 📌 Conclusion

Ce projet montre qu’il est possible de :

* prédire efficacement l’acceptation des Pull Requests IA,
* comprendre les décisions du modèle grâce à SHAP,
* relativiser l’impact des agents IA par rapport aux facteurs humains et techniques.

👉 Les agents IA influencent les PR, mais ce sont surtout la qualité, la taille et la collaboration humaine qui déterminent leur acceptation.

---

## 👤 Auteur

**Thierry Kouadio**
Maîtrise en génie logiciel – ÉTS Montréal
Projet final – AIDev / Mining Software Repositories

