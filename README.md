# Mini-projet Data Mining — détection des crises d'absence en EEG

Chaîne complète allant du signal EEG brut à un modèle capable de détecter les périodes
de crise d'absence.

## Organisation

```
data/raw/        fichiers fournis : <id>d.mat (signal), <id>h.mat (marqueurs),
                 Crises absences.xlsx (annotations des crises)
notebooks/       main.ipynb — le rendu : sections, appels aux fonctions, résultats
src/             le code réutilisable, appelé depuis le notebook
reports/         rapport et figures exportées
```

| Module | Contenu | Partie du sujet |
| --- | --- | --- |
| `src/data.py` | lecture des fichiers MATLAB, marqueurs, annotations, résumés | I et II |
| `src/preprocessing.py` | filtrage (coupe-bande secteur, passe-bande 0,5–40 Hz), puissances par bande | prétraitement |
| `src/visualization.py` | signal, montage multi-canaux, spectre, marqueurs | I |
| `src/labeling.py` | variable cible à partir des annotations | III |
| `src/windowing.py` | découpage en fenêtres de 2 s | IV |
| `src/features.py` | caractéristiques temporelles et fréquentielles | V et VI |
| `src/training.py` | Decision Tree, Random Forest, KNN, SVM | VIII |
| `src/evaluation.py` | accuracy, précision, rappel, F1, matrice de confusion | IX |
| `src/detection.py` | reconstruction des intervalles de crise | X |

Les modules des parties III à X sont encore vides : ils seront remplis au fur et à mesure.

## Installation

```bash
pip install numpy pandas scipy matplotlib openpyxl scikit-learn
jupyter lab            # puis ouvrir notebooks/main.ipynb
```

Le notebook retrouve seul la racine du projet, il peut être lancé depuis n'importe
quel dossier du dépôt.

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

## Point ouvert

Les annotations de `Crises absences.xlsx` sont des **heures de la journée** (fractions de
journée Excel), alors que le signal démarre à `t = 0 s`. L'heure de début d'enregistrement
n'est pas stockée dans les fichiers `.mat` : il manque une origine commune pour convertir
les annotations en secondes depuis le début du tracé. À confirmer avec l'encadrant avant
la Partie III.

## Avancement

- [x] Partie I — exploration, questions 1 et 2
- [ ] Partie II — exploitation des annotations (questions 3 et 4)
- [ ] Parties III à XII
