# Détection automatique des crises d'absence à partir de signaux EEG

Rapport intermédiaire : Parties I à IX (Questions 1 à 12). Le détail du code et toutes les
sorties sont dans `notebooks/main.ipynb` ; les figures sont dans `reports/figures/`.

## 1. Introduction

Une crise d'absence est une courte perte de conscience (quelques secondes) qui se voit sur
l'EEG comme une décharge généralisée de pointes-ondes vers 3 Hz. Le but est de construire une
chaîne de Data Mining qui part du signal EEG brut et produit un modèle capable de dire, pour
chaque fenêtre de 2 s, s'il s'agit d'une crise.

## 2. Données

* 22 enregistrements EEG au format MATLAB (`timetable`), 256 Hz, 19 électrodes du système 10-20
  plus des voies annexes (ECG, SLI, EMG), de 20 à 60 minutes chacun.
* Annotations : `Annotations_MAJ.xlsx`. Les débuts et fins de crise sont des heures de la
  journée ; la colonne `DebutTrace` donne l'heure de début du tracé, d'où
  `début (s) = DebCEk − DebutTrace` et `durée = FinCEk − DebCEk`.
* Contrôle qualité : deux enregistrements en double (première ligne gardée), une faute de saisie
  corrigée (`210504B-C`, crise de 6 h ramenée à 8 s d'après l'ancien fichier), trois
  enregistrements sans annotation exclus. Il reste **19 enregistrements, 83 crises, 8,1 h de signal**.
* L'alignement a été vérifié crise par crise : l'amplitude (RMS) pendant une crise vaut 3 à 10
  fois celle hors crise du même patient. Sur `230102B_F`, `DebutTrace` confirme à 0,25 s près
  l'origine estimée en Partie I à partir du seul signal.

## 3. Méthodologie

| Étape | Choix | Raison |
| --- | --- | --- |
| Prétraitement | coupe-bande 50 Hz + passe-bande 0,5–40 Hz, sans déphasage (`filtfilt`) | le signal brut est dominé par le secteur |
| Variable cible | `Class = 1` si `début ≤ t ≤ fin` | consigne |
| Segmentation | fenêtres de 2 s (512 échantillons), recouvrement 50 % ; classe 1 si ≥ 50 % des échantillons en crise | consigne |
| Caractéristiques | Mean, Std, Variance, Min, Max, Amplitude, RMS, Energy ; puissances Delta, Theta, Alpha, Beta par FFT (Hann) | consigne |
| Multi-canal | chaque caractéristique est calculée sur les 19 canaux puis on garde la **médiane** | une absence touche tous les canaux, un artefact quelques-uns |
| Séparation | 80 % / 20 % stratifiée, `random_state = 42` | consigne |
| Réglage | `GridSearchCV`, validation croisée stratifiée à 5 plis, critère **F1** de la classe crise | l'accuracy est trompeuse avec 4 % de crises |
| KNN et SVM | log10 des variables positives puis standardisation | gain de F1 de 0,03 à 0,05 en validation croisée |

## 4. Résultats

### Crises (Questions 3 et 4)

* Durée : moyenne 13,3 s, médiane 11 s, de 2 à 60 s ; coefficient de variation 78 %.
  Les crises n'ont **pas** une durée similaire : la durée dépend surtout du patient.
* Intervalle moyen entre deux crises (fin → début suivant) : 193 s, de 52 s à 13 min selon le patient.
* `230102B_F` : 9 crises de 8 à 24 s (moyenne 15 s), intervalle moyen 128 s.

### Déséquilibre (Questions 6 et 8)

3,8 % des échantillons et **4,1 % des fenêtres** sont en crise (28 994 fenêtres, dont 1 182 de
classe 1). Un modèle qui répond toujours « pas de crise » obtient 95,9 % d'accuracy.

### Caractéristiques (Questions 9 à 11)

* En crise, la médiane de l'écart-type est multipliée par 6, celle de l'énergie par 40, Delta
  par 50, Theta par 19. La part de Delta dans la puissance passe de 68 % à 80 %.
* Redondances : Std = RMS (r = 1,00), Variance = Energy (r = 1,00), Variance = Std² (Spearman 1,00),
  Amplitude très liée à Std (r = 0,97). Theta, Alpha et Beta sont liées entre elles (r de 0,79 à 0,91).
* Pouvoir discriminant (AUC univariée) : Amplitude 0,94, puis Max, Min, Theta, Std, RMS (0,93) ;
  Mean est inutile (0,51) car le filtrage retire la composante continue.

### Modèles (Question 12, jeu de test)

| Modèle | Accuracy | Precision | Recall | F1 | Hyperparamètres retenus |
| --- | --- | --- | --- | --- | --- |
| **Random Forest** | 0,984 | 0,872 | 0,720 | **0,789** | 300 arbres, min_samples_leaf = 1 |
| Gradient Boosting (proposé) | 0,982 | 0,830 | 0,703 | 0,761 | learning_rate 0,05, 31 feuilles |
| SVM (RBF) | 0,982 | 0,838 | 0,678 | 0,749 | C = 10, gamma = 0,3 |
| Decision Tree (Entropie) | 0,980 | 0,808 | 0,678 | 0,737 | profondeur 8, min_samples_leaf = 20 |
| KNN | 0,980 | 0,784 | 0,691 | 0,734 | k = 11, pondéré par la distance |
| Decision Tree (Gini) | 0,979 | 0,782 | 0,682 | 0,729 | profondeur 6, min_samples_leaf = 20 |
| Référence « toujours 0 » | 0,959 | 0 | 0 | 0 | |

**Gini contre entropie.** À profondeur égale, les deux critères donnent le même F1 (écarts de
0,01 à 0,03, du même ordre que la variation entre plis). Les arbres optimisés ont un F1 de 0,73
et 0,74 sur le test et prennent la même décision pour 99,3 % des fenêtres. L'arbre Gini est deux
fois plus petit (41 feuilles contre 82) : il est retenu. Les deux arbres posent d'abord la
question « Beta ≤ 191 µV² », qui sépare les pointes rapides des artefacts lents.

**Généralisation.** Avec une validation croisée par enregistrement (aucune fenêtre d'un même
patient à la fois en entraînement et en test), le F1 descend à 0,70 (KNN), 0,70 (SVM),
0,69 (Random Forest), 0,66 (Gradient Boosting) et 0,52 à 0,58 (arbres seuls).

## 5. Discussion

* L'accuracy ne départage pas les modèles (97,9 % à 98,4 %) et la référence naïve atteint
  95,9 % : il faut juger sur Precision, Recall et F1.
* Tous les modèles ont une Precision plus haute que le Recall : peu de fausses alarmes, mais
  environ 30 % des fenêtres de crise sont manquées (surtout les bords de crise et les crises de 2 à 3 s).
* La séparation aléatoire est optimiste : deux fenêtres voisines partagent 50 % de leurs
  échantillons. La performance réaliste sur un nouveau patient est d'environ F1 = 0,70.
* Les différences entre patients pèsent lourd : `230718A-A` a un niveau de base 4 fois plus
  élevé que les autres, ce qui piège les seuils d'amplitude appris ailleurs.

## 6. Suite prévue

Question 13 (choix de la métrique), détection temporelle des crises (Partie X), analyse des
erreurs (Partie XI) et améliorations (Partie XII) : normalisation par enregistrement, puissances
relatives, sélection des variables non redondantes, seuil de décision réglé pour le Recall,
lissage temporel des prédictions.
