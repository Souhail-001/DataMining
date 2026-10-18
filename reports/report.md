# Détection automatique des crises d'absence à partir de signaux EEG

Rapport final : Parties I à XII (Questions 1 à 14). Le détail du code et toutes les sorties sont
dans `notebooks/main.ipynb` ; les figures sont dans `reports/figures/`.

## 1. Introduction

Une crise d'absence est une courte perte de conscience (quelques secondes) qui se voit sur
l'EEG comme une décharge généralisée de pointes-ondes vers 3 Hz. Le but est de construire une
chaîne de Data Mining qui part du signal EEG brut et produit un modèle capable de dire, pour
chaque fenêtre de 2 s, s'il s'agit d'une crise, puis de retrouver les crises dans le temps
(début et fin) et de les comparer aux annotations du médecin.

Pipeline réalisé : signal EEG → exploration → prétraitement (filtrage) → segmentation en fenêtres
→ extraction de caractéristiques → dataset de classification → modèles (arbre, forêt, KNN, SVM,
Gradient Boosting) → évaluation → détection temporelle → analyse des erreurs → améliorations.

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
| Généralisation | validation croisée **par enregistrement** (`GroupKFold`, 5 plis) | mesure réaliste sur un patient jamais vu |
| Détection temporelle | chaque enregistrement prédit par un modèle entraîné sur les autres ; fenêtres consécutives regroupées en intervalles | toutes les fenêtres deviennent des fenêtres de test |
| Amélioration | seuil choisi en validation interne, lissage sur 3 fenêtres, règles de durée | issues de l'analyse des erreurs |

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
| **Random Forest** | 0,984 | 0,872 | 0,720 | **0,789** | 200 arbres, min_samples_leaf = 1 |
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

### Choix de la métrique (Question 13)

| Modèle | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Toujours 0 | 0,959 | 0 | 0 | 0 |
| Toujours 1 | 0,041 | 0,041 | 1 | 0,078 |
| Au hasard (4 % de 1) | 0,922 | 0,05 | 0,05 | 0,05 |
| Random Forest (test 80/20) | 0,984 | 0,872 | 0,720 | 0,789 |

L'accuracy est dominée par la classe majoritaire : « toujours 0 » obtient 95,9 % sans trouver
une seule crise. Le **Recall** est prioritaire (rater une crise est l'erreur la plus grave), mais
il se triche seul (« toujours 1 » a un Recall de 1) : on le contrôle par la **Precision** et on
résume par le **F1-score**, critère de réglage de tous les modèles. Pour l'usage réel, on ajoute
deux mesures par crise : la part des crises détectées et les fausses alarmes par heure.

### Détection temporelle (Partie X, Question 14)

Chaque enregistrement est prédit par une Random Forest entraînée sur les autres (le test 80/20
ne garde qu'une fenêtre sur cinq par enregistrement et ne permet pas de reconstruire une crise
continue). Les fenêtres consécutives prédites crise forment un intervalle ; chaque fenêtre
représente sa seconde centrale.

Exemple `230102B_F` (seuil 0,5) : les 9 crises sont détectées, début détecté 0,5 à 1,5 s après
le début annoté, fin à ±0,5 s, 60 à 97 % de chaque crise couverte. La crise 7 (24 s) est
coupée en 3 morceaux et il y a 8 fausses alarmes de 1 à 2 s.

Sur les 19 enregistrements : **81 crises sur 83 détectées (97,6 %)**, mais **10,8 fausses
alarmes par heure**, concentrées sur `210427B-C` et `210504B-C`.

### Analyse des erreurs (Partie XI)

| Type | Fenêtres |
| --- | --- |
| Vrais positifs | 753 |
| Faux négatifs | 429 |
| Faux positifs | 240 |
| Vrais négatifs | 27 572 |

* **Faux négatifs** : surtout aux bords des crises (73 % des fenêtres à 50–75 % en crise sont
  manquées, contre 30 % au cœur), dans les crises longues dont l'amplitude baisse au milieu, et
  chez deux patients (`191113A-D`, `220106A-A`) dont les crises sont peu amples par rapport à leur
  activité normale. Exemple : au milieu de la crise 7 de `230102B_F`, un grand artefact de
  mouvement fait répondre « hors crise » au modèle.
* **Faux positifs** : 96 % loin de toute crise, très courts (1 à 2 fenêtres), et la moitié sur
  deux enregistrements riches en artefacts. Leur amplitude est proche de celle d'une crise : les
  variables mesurent surtout « combien », pas la forme du signal (rythme à 3 Hz).

### Améliorations (Partie XII)

Toutes les variantes sont évaluées par enregistrement (patients jamais vus). Le seuil est choisi
par une validation interne sur les seuls enregistrements d'entraînement de chaque pli.

| Variante | Precision | Recall | F1 | Crises détectées | Fausses alarmes / h |
| --- | --- | --- | --- | --- | --- |
| Base : Random Forest, seuil 0,5 | 0,76 | 0,64 | 0,69 | 81/83 | 10,8 |
| A. + contexte temporel (moyennes des fenêtres voisines) | 0,73 | 0,70 | 0,72 | 78/83 | 7,3 |
| B. + normalisation par enregistrement | 0,74 | 0,64 | 0,68 | 80/83 | 9,8 |
| C. + poids des classes équilibrés | 0,70 | 0,70 | 0,70 | 82/83 | 16,5 |
| D. seuil choisi en validation interne | 0,71 | 0,69 | 0,70 | 81/83 | 16,1 |
| E. seuil + lissage sur 3 fenêtres | 0,73 | 0,74 | 0,74 | 81/83 | 6,8 |
| **F. E + règles de durée (retenu)** | **0,74** | **0,75** | **0,74** | **81/83** | **5,2** |

* **Retenu** : seuil (≈ 0,30–0,35) + lissage + règles de durée (combler un trou d'une fenêtre,
  supprimer une détection d'une seule fenêtre, plus courte que la plus courte crise). F1 de
  0,69 à 0,74, Recall de 0,64 à 0,75, fausses alarmes divisées par 2, aucune crise perdue.
* Le contexte temporel aide aussi (F1 0,72) mais perd 3 crises courtes ; le lissage fait mieux,
  plus simplement.
* **Rejetés** : la normalisation par enregistrement (on perd l'amplitude absolue, très
  informative) et le rééquilibrage des classes (fausses alarmes ×1,5 pour le même F1).

## 5. Discussion

* L'accuracy ne départage pas les modèles (97,9 % à 98,4 %) et la référence naïve atteint
  95,9 % : il faut juger sur Precision, Recall et F1, et pour la détection, par crise.
* La séparation aléatoire 80/20 est optimiste : deux fenêtres voisines partagent 50 % de leurs
  échantillons. Sur un nouveau patient, le F1 réaliste est d'environ 0,70, et 0,74 après
  amélioration.
* Le Recall par fenêtre sous-estime la qualité de la détection : on manque surtout des
  *morceaux* de crises (bords, milieu des crises longues), presque jamais des crises entières.
* Le point faible est la **spécificité** : des artefacts brefs et amples déclenchent des
  alarmes. Le lissage et la durée minimale les réduisent de moitié, sans les supprimer.
* Les différences entre patients pèsent lourd : `230718A-A` a un niveau de base 4 fois plus
  élevé que les autres, `191113A-D` a des crises très peu amples. Un modèle appris sur
  18 patients ne couvre pas toute cette variabilité.

## 6. Conclusion

**Bilan.** La chaîne complète a été construite et évaluée : lecture des fichiers MATLAB, contrôle
et correction des annotations, filtrage, fenêtres de 2 s, 12 caractéristiques temporelles et
fréquentielles, cinq modèles comparés. La Random Forest est la meilleure (F1 = 0,79 sur le test
80/20, 0,69 sur un nouveau patient). Après seuil, lissage et règles de durée, le système détecte
**81 crises sur 83 chez des patients jamais vus**, avec environ 1 s de retard au début et
**5 fausses alarmes par heure**.

**Limites.**

* 19 enregistrements seulement, dont quelques-uns très différents des autres : la performance
  varie beaucoup d'un patient à l'autre.
* Les annotations sont à la seconde près et ont demandé des corrections ; les bords de crise
  restent incertains.
* Les caractéristiques mesurent l'énergie du signal, pas sa forme : un artefact ample ressemble
  à une crise. Il faudrait des variables de rythmicité (pointe-onde à 3 Hz), des fenêtres plus
  courtes pour mieux placer le début, ou un réseau de neurones sur le signal brut, avec plus de
  données.
* 5 fausses alarmes par heure restent trop pour une surveillance clinique sans relecture humaine :
  le système est adapté comme aide au repérage des crises dans un long enregistrement.
