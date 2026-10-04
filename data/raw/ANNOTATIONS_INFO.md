# Annotations des crises : fichier à utiliser

**Fichier de référence : `Annotations_MAJ.xlsx`** (reçu le 1er octobre 2026).
L'ancien fichier `Crises absences.xlsx` est gardé pour traçabilité mais n'est plus utilisé.

## Message de l'encadrante (Ines Bouzouita, 1er octobre 2026, à 3AINFO1 et 3AINFO2)

> Suite à notre discussion concernant les données, une nouvelle colonne a été ajoutée au
> fichier annotationsMAJ.xls. Cette colonne indique désormais l'heure de début de
> l'enregistrement du tracé.
>
> Nous pouvons ainsi déterminer correctement la durée de chaque crise à partir des heures de
> début et de fin indiquées dans les annotations.
>
> Nous allons donc considérer les valeurs de début et de fin de crise comme des horaires, et
> non comme des durées.
>
> Par exemple, pour l'identifiant 190304A-E :
> début de l'enregistrement du tracé : 10:13:35 ;
> début de la crise 1 : 10:35:06 ; fin de la crise 1 : 10:35:13 ;
> durée de la crise 1 : 10:35:13 − 10:35:06 = 7 secondes.

## Règle appliquée dans le code (`src/data.py`, `load_seizure_annotations`)

Les cellules Excel sont des fractions de journée (`0,5` = 12:00:00). Multipliées par 86 400,
elles donnent des secondes depuis minuit.

| Grandeur | Calcul |
| --- | --- |
| début de la crise k (s depuis le début du tracé) | `DebCEk − DebutTrace` |
| fin de la crise k (s depuis le début du tracé) | `FinCEk − DebutTrace` |
| durée de la crise k | `FinCEk − DebCEk` |

## Points relevés dans `Annotations_MAJ.xlsx` et décisions

| Enregistrement | Problème | Décision |
| --- | --- | --- |
| 230406B-A | présent 2 fois (lignes 11 et 19), heures différentes de 6 s au plus | première ligne gardée (identique à l'ancien fichier) |
| 230515B-G | présent 2 fois (lignes 12 et 20), heures différentes de 11 s au plus | première ligne gardée |
| 210504B-C | `FinCE1 = 0,785486`, soit une crise de 6 h | corrigé en `0,535486` (8 s), valeur de l'ancien fichier |
| 210914B-A | `NombreCE = 6` mais 11 intervalles remplis (absences « atypiques ») | les 11 intervalles sont gardés |
| 200625A_F, 210204B_C, 210208B_G | fichiers `.mat` présents mais absents du fichier d'annotations | exclus de l'apprentissage |
| 210504B-C, crise 12 | annotée 60 s, mais la décharge visible dure environ 20 s | gardée telle quelle (même valeur dans l'ancien fichier) |

Résultat : 19 enregistrements, 83 crises. Sur `230102B_F`, `DebutTrace` (12:04:53) confirme à
0,25 s près l'origine estimée en Partie I à partir du signal.
