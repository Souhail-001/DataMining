"""Code du mini-projet de Data Mining : detection des crises d'absence en EEG.

Modules
-------
data           chargement des fichiers MATLAB, des marqueurs et des annotations
visualization  figures (signal, montage multi-canaux, marqueurs)
windowing      decoupage du signal en fenetres (Partie IV)
features       caracteristiques temporelles et frequentielles (Parties V et VI)
labeling       construction de la variable cible (Partie III)
training       entrainement des modeles (Partie VIII)
evaluation     metriques et matrices de confusion (Partie IX)
detection      reconstruction des intervalles de crise (Partie X)
"""

__version__ = "0.1.0"
