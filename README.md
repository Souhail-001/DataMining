# Mini-projet Data Mining — détection des crises d'absence en EEG

Chaîne complète allant du signal EEG brut à un modèle capable de détecter les périodes
de crise d'absence.

## Organisation

```
data/raw/        fichiers fournis : <id>d.mat (signal), <id>h.mat (marqueurs),
                 Annotations_MAJ.xlsx (annotations des crises, version du 1er octobre)
                 ANNOTATIONS_INFO.md (regle de conversion et corrections appliquees)
data/processed/  windows_features.csv : le dataset de fenetres (genere par le notebook)
notebooks/       main.ipynb — le rendu : sections, appels aux fonctions, résultats
src/             le code réutilisable, appelé depuis le notebook
app.py           interface Streamlit d'exploration (mêmes fonctions que le notebook)
ui/              interface web interactive : signaux, modèles, arbres (python ui/serve.py)
scripts/         export_ui.py : prépare les données de l'interface web
reports/         rapport et figures exportées
```

| Module | Contenu | Partie du sujet |
| --- | --- | --- |
| `src/data.py` | lecture des fichiers MATLAB, marqueurs, annotations, résumés | I et II |
| `src/preprocessing.py` | filtrage (coupe-bande secteur, passe-bande 0,5–40 Hz), puissances par bande | prétraitement |
| `src/visualization.py` | signal, montage multi-canaux, spectre, marqueurs | I |
| `src/labeling.py` | variable cible à partir des annotations | III |
| `src/windowing.py` | découpage en fenêtres de 2 s | IV |
| `src/features.py` | caractéristiques temporelles et puissances par bande (FFT), dataset complet | V et VI |
| `src/exploration.py` | statistiques des crises, comparaison des classes, corrélations | II et VII |
| `src/training.py` | séparation 80/20, Decision Tree (Gini et entropie), Random Forest, KNN, SVM, Gradient Boosting | VIII |
| `src/evaluation.py` | accuracy, précision, rappel, F1, matrices de confusion, test par enregistrement | IX |
| `src/detection.py` | reconstruction des intervalles de crise | X (à faire) |

## Installation

```bash
pip install numpy pandas scipy matplotlib openpyxl scikit-learn streamlit
jupyter lab            # puis ouvrir notebooks/main.ipynb
```

Le notebook retrouve seul la racine du projet, il peut être lancé depuis n'importe
quel dossier du dépôt.

## Interface d'exploration

```bash
streamlit run app.py   # depuis la racine du projet
```

`app.py` n'est qu'une couche d'affichage : elle appelle les mêmes fonctions que le
notebook, donc une figure vue dans l'interface est exactement celle du rapport.
Sept onglets : aperçu, signal, montage multi-canaux, spectre, puissances par bande,
qualité des données, marqueurs et annotations.

La barre latérale choisit l'enregistrement, le canal et le prétraitement (coupe-bande
secteur détecté automatiquement, passe-bande réglable) ; chaque onglet se cale sur une
fenêtre temporelle et peut sauter directement à un marqueur du manipulateur. La case
« Proposer le téléchargement des figures » ajoute un bouton PNG sous chaque figure ;
elle est décochée par défaut car l'encodage coûte environ 0,3 s par figure et par
rafraîchissement.

`.streamlit/config.toml` force le thème clair : les figures matplotlib du projet sont
dessinées sur fond blanc et deviendraient illisibles en thème sombre.

## Interface web interactive (`ui/`)

Station de lecture EEG dans le navigateur : tracé 19 voies avec les crises annotées et les
prédictions des modèles, carte du scalp et spectre FFT de chaque fenêtre de 2 s, laboratoire
du seuil de décision (Precision/Recall/F1 en direct), arbres Gini et entropie côte à côte avec
le chemin de décision d'une fenêtre, nuage de points des caractéristiques.

```bash
python ui/serve.py              # ouvre http://127.0.0.1:8765 dans le navigateur
python scripts/export_ui.py     # régénère ui/data/ (environ 10 minutes)
```

Ouvrir `ui/index.html` par un double-clic ne marche pas : le navigateur bloque la lecture de
`ui/data/` sans serveur. Les probabilités affichées sur le tracé sont obtenues **par patient**
(GroupKFold) : chaque enregistrement est prédit par un modèle entraîné sans lui.
Le signal est stocké filtré, à 128 Hz, en blocs de 300 s (`ui/data/sig/<id>/<k>.png` : des
entiers 16 bits rangés sans perte dans des images PNG en niveaux de gris).

## Format des données

Les fichiers `.mat` contiennent un `timetable` MATLAB enregistré comme objet (MCOS).
`scipy.io.loadmat` ne sait pas le décoder : les valeurs sont dans la variable cachée
`__function_workspace__`. `src/data.py` relit cet espace, retrouve la structure par ses
champs, concatène les blocs d'une seconde et reconstruit un `DataFrame`.

Sur `230102B_F_0000` : 313 600 lignes, 22 colonnes (temps + 19 canaux EEG + SLI + ECG),
256 Hz, 1 225 s, aucune valeur manquante.

Deux points relevés à l'exploration :

* le signal brut est dominé par le 50 Hz du secteur — le filtrage n'est pas optionnel ;
* les canaux `ECG`, `Fz` saturent par moments et `SLI` est constant : ils sont écartés.

## Annotations

Le fichier `Annotations_MAJ.xlsx` donne l'heure de début du tracé (`DebutTrace`). Les heures
des crises sont converties en secondes depuis le début du tracé : `DebCEk − DebutTrace`.
Deux doublons et une faute de saisie sont traités dans le code ; le détail est dans
`data/raw/ANNOTATIONS_INFO.md`. Résultat : 19 enregistrements annotés, 83 crises.

## Résultats (Questions 1 à 12)

Dataset : 28 994 fenêtres de 2 s (recouvrement 50 %), 12 caractéristiques, 4,1 % de fenêtres en crise.
Hyperparamètres choisis par validation croisée à 5 plis sur le F1 ; test sur 20 % stratifié.

| Modèle | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Random Forest | 0,984 | 0,872 | 0,720 | **0,789** |
| Gradient Boosting (proposé) | 0,982 | 0,830 | 0,703 | 0,761 |
| SVM (RBF) | 0,982 | 0,838 | 0,678 | 0,749 |
| Decision Tree (Entropie) | 0,980 | 0,808 | 0,678 | 0,737 |
| KNN | 0,980 | 0,784 | 0,691 | 0,734 |
| Decision Tree (Gini) | 0,979 | 0,782 | 0,682 | 0,729 |
| Référence « toujours 0 » | 0,959 | 0 | 0 | 0 |

Gini et entropie donnent des arbres équivalents ; Gini est retenu (2 fois moins de feuilles).
Sur des patients jamais vus (validation croisée par enregistrement), le F1 descend à
environ 0,70 pour KNN, SVM et Random Forest.

## Avancement

- [x] Partie I : exploration (Q1, Q2)
- [x] Partie II : annotations (Q3, Q4)
- [x] Parties III à VII : classe, fenêtres, caractéristiques, FFT, analyse exploratoire (Q5 à Q11)
- [x] Parties VIII et IX : modèles et évaluation (Q12)
- [ ] Question 13 et Parties X à XII : choix de la métrique, détection temporelle, erreurs, améliorations
