"""Chargement des enregistrements EEG du mini-projet (Partie I).

Les fichiers `*d.mat` contiennent un `timetable` MATLAB. Ce type est enregistre
comme un objet MCOS : `scipy.io.loadmat` ne sait pas le decoder et renvoie un
`MatlabOpaque` vide. Les valeurs reelles sont stockees dans la variable cachee
`__function_workspace__`, qui est elle-meme un mini fichier MAT.

Ce module relit cet espace de travail, reconstruit le signal sous forme de
`pandas.DataFrame` et expose un objet `Recording` utilise par le notebook.

Conventions MATLAB utiles :
  * un `timetable` decoupe le signal en blocs (ici 1 bloc = 1 seconde) ;
  * un `duration` est stocke en millisecondes ;
  * un tableau `string` est stocke sous forme d'entiers 64 bits.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.io import loadmat

try:  # SciPy >= 1.8
    from scipy.io.matlab._mio5 import MatFile5Reader
except ImportError:  # pragma: no cover - SciPy plus ancien
    from scipy.io.matlab.mio5 import MatFile5Reader


__all__ = [
    "Recording",
    "find_project_root",
    "list_recordings",
    "load_recording",
    "load_events",
    "load_seizure_annotations",
    "summarize",
    "channel_quality",
    "saturation_report",
]

# Champs presents dans la structure interne d'un timetable MATLAB.
_TIMETABLE_FIELDS = {"data", "varNames", "numRows", "numVars", "rowTimes"}

# Un duration MATLAB est stocke en millisecondes.
_MS_PER_SECOND = 1000.0

# Prefixes des canaux reellement EEG (les autres sont SLI, ECG, ...).
_EEG_PREFIX = "EEG"


# ---------------------------------------------------------------------------
# Localisation des fichiers
# ---------------------------------------------------------------------------
def find_project_root(start: Path | str | None = None) -> Path:
    """Remonte l'arborescence jusqu'au dossier contenant `data/raw`."""
    current = Path(start or Path.cwd()).resolve()

    for folder in [current, *current.parents]:
        if (folder / "data" / "raw").is_dir():
            return folder

    raise FileNotFoundError(
        "Dossier `data/raw` introuvable. Lancez Jupyter depuis le dossier du projet."
    )


def raw_dir(root: Path | str | None = None) -> Path:
    """Chemin du dossier des donnees brutes."""
    return find_project_root(root) / "data" / "raw"


def list_recordings(root: Path | str | None = None) -> pd.DataFrame:
    """Inventaire des enregistrements disponibles (un couple `d`/`h` par ligne)."""
    folder = raw_dir(root)

    rows = []
    for data_path in sorted(folder.glob("*d.mat")):
        recording_id = data_path.stem[:-1]
        header_path = folder / f"{recording_id}h.mat"

        rows.append(
            {
                "recording_id": recording_id,
                "data_file": data_path.name,
                "header_file": header_path.name if header_path.is_file() else None,
                "size_mb": round(data_path.stat().st_size / 1024**2, 2),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["recording_id", "data_file", "header_file", "size_mb"],
    )


# ---------------------------------------------------------------------------
# Lecture bas niveau du stockage objet MATLAB
# ---------------------------------------------------------------------------
def _read_object_pool(path: Path) -> np.ndarray:
    """Retourne les elements stockes dans l'espace objet MCOS du fichier."""
    contents = loadmat(str(path))
    workspace = contents.get("__function_workspace__")

    if workspace is None:
        raise ValueError(
            f"{path.name} ne contient pas de stockage objet MATLAB "
            "(`__function_workspace__` absent)."
        )

    raw = workspace.tobytes()

    if raw[2:4] != b"IM":
        raise ValueError(f"{path.name} : seul l'ordre d'octets little-endian est gere.")

    stream = io.BytesIO(raw)
    reader = MatFile5Reader(stream, byte_order="<")
    reader.initialize_read()

    stream.seek(8)  # on saute l'entete du mini fichier MAT embarque
    header, _ = reader.read_var_header()
    root = reader.read_var_array(header)

    if "MCOS" not in (root.dtype.names or ()):
        raise ValueError(f"{path.name} : stockage objet MCOS introuvable.")

    wrapper = root["MCOS"][0, 0]
    payload_field = wrapper.dtype.names[-1]  # le nom varie selon la version de SciPy

    return np.asarray(wrapper[payload_field][0]).ravel()


def _find_timetable(pool: Sequence[np.ndarray]) -> np.void:
    """Cherche dans l'espace objet la structure decrivant le timetable.

    On identifie la structure par ses champs et non par sa position : l'ordre
    des elements depend du fichier et de la version de MATLAB.
    """
    for element in pool:
        element = np.asarray(element)
        if _TIMETABLE_FIELDS.issubset(set(element.dtype.names or ())) and element.size:
            return element.ravel()[0]

    raise ValueError("Aucune structure de type timetable trouvee dans le fichier.")


def _find_row_times_ms(pool: Sequence[np.ndarray], n_rows: int) -> np.ndarray:
    """Cherche le vecteur des instants (duration MATLAB, en millisecondes).

    Le champ `rowTimes` du timetable n'est qu'une reference vers un autre
    element de l'espace objet. On retrouve cet element par sa taille et par le
    fait qu'il est fini et strictement croissant.
    """
    for element in pool:
        element = np.asarray(element)

        if element.dtype.names or element.dtype.kind != "f":
            continue
        if element.size != n_rows or element.ndim > 2:
            continue
        if element.ndim == 2 and min(element.shape) != 1:
            continue

        values = element.ravel().astype(float)

        if not np.isfinite(values).all():
            continue
        if n_rows > 1 and not np.all(np.diff(values) > 0):
            continue

        return values

    raise ValueError("Vecteur des instants (rowTimes) introuvable dans le fichier.")


def _clean_label(text: str) -> str:
    """Repare les accents perdus a l'ecriture du fichier MATLAB.

    Les libelles accentues ont ete enregistres avec deux caracteres de
    remplacement U+FFFD a la place de l'octet UTF-8 d'origine ("D\ufffd\ufffdbut").
    Le seul accent utilise dans ces marqueurs est le "e" accent aigu.
    """
    return text.replace("\ufffd\ufffd", "\u00e9")


def _decode_string_array(packed: np.ndarray) -> list[str]:
    """Decode un tableau `string` MATLAB stocke en entiers 64 bits.

    Disposition : [version, nDims, dims..., longueurs..., texte UTF-16].
    Les longueurs comptent des unites de 16 bits, pas des octets.
    """
    values = np.asarray(packed).ravel().astype(np.uint64)

    if values.size < 4:
        raise ValueError("Tableau de chaines trop court pour etre decode.")

    n_dims = int(values[1])
    dims = values[2 : 2 + n_dims].astype(int)
    count = int(np.prod(dims))

    offset = 2 + n_dims
    lengths = values[offset : offset + count].astype(int)
    payload = values[offset + count :]

    if count == 0 or lengths.sum() * 2 > payload.size * 8:
        raise ValueError("Longueurs de chaines incoherentes avec le contenu.")

    text = payload.astype("<u8").tobytes().decode("utf-16-le", "replace")

    strings, position = [], 0
    for length in lengths:
        strings.append(_clean_label(text[position : position + length]))
        position += length

    return strings


def _find_string_array(pool: Sequence[np.ndarray], count: int) -> list[str] | None:
    """Cherche un tableau `string` contenant exactement `count` elements."""
    for element in pool:
        element = np.asarray(element)

        if element.dtype.names or element.dtype.kind != "u" or element.size < 4:
            continue

        try:
            strings = _decode_string_array(element)
        except (ValueError, UnicodeError):
            continue

        if len(strings) == count:
            return strings

    return None


# ---------------------------------------------------------------------------
# Reconstruction du signal
# ---------------------------------------------------------------------------
def _channel_names(table: np.void) -> list[str]:
    return [str(np.asarray(name).item()) for name in np.asarray(table["varNames"]).ravel()]


def _stack_channels(table: np.void) -> tuple[np.ndarray, int, int]:
    """Concatene les blocs de chaque canal. Retourne (signal, blocs, taille bloc)."""
    n_blocks = int(np.asarray(table["numRows"]).item())
    n_channels = int(np.asarray(table["numVars"]).item())

    columns = np.asarray(table["data"]).ravel()
    names = _channel_names(table)

    if len(columns) != n_channels or len(names) != n_channels:
        raise ValueError("Le nombre de canaux ne correspond pas aux metadonnees.")

    blocks_first = np.asarray(columns[0]).ravel()
    samples_per_block = np.asarray(blocks_first[0]).size

    channels = []
    for name, column in zip(names, columns):
        blocks = np.asarray(column).ravel()

        if len(blocks) != n_blocks:
            raise ValueError(f"Nombre de blocs inattendu pour le canal {name}.")

        sizes = {np.asarray(block).size for block in blocks}
        if sizes != {samples_per_block}:
            raise ValueError(f"Blocs de tailles inegales pour le canal {name}.")

        channels.append(
            np.concatenate([np.asarray(block, dtype=float).ravel() for block in blocks])
        )

    return np.column_stack(channels), n_blocks, samples_per_block


def _sampling_rate(row_times_s: np.ndarray, samples_per_block: int) -> float:
    """Frequence d'echantillonnage deduite de la duree d'un bloc."""
    if row_times_s.size < 2:
        raise ValueError(
            "Un seul bloc temporel : la frequence d'echantillonnage ne peut pas "
            "etre deduite."
        )

    steps = np.diff(row_times_s)
    step = float(np.median(steps))

    if not np.allclose(steps, step, rtol=1e-6, atol=1e-9):
        raise ValueError(
            "Blocs temporels irreguliers : l'enregistrement contient des coupures."
        )

    return samples_per_block / step


# ---------------------------------------------------------------------------
# Objet principal
# ---------------------------------------------------------------------------
@dataclass
class Recording:
    """Un enregistrement EEG charge en memoire.

    Attributs
    ---------
    data : DataFrame avec une colonne de temps puis un canal par colonne.
    sampling_rate : frequence d'echantillonnage en Hz.
    events : marqueurs du manipulateur (YO, YF, HPN, SLI, saturations).
    """

    recording_id: str
    data: pd.DataFrame
    sampling_rate: float
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    time_column: str = "Time"
    amplitude_unit: str | None = None
    source: dict = field(default_factory=dict)

    # -- dimensions ---------------------------------------------------------
    @property
    def n_samples(self) -> int:
        """Nombre de lignes du fichier."""
        return len(self.data)

    @property
    def n_variables(self) -> int:
        """Nombre de colonnes, colonne de temps comprise."""
        return self.data.shape[1]

    @property
    def channels(self) -> list[str]:
        return [name for name in self.data.columns if name != self.time_column]

    @property
    def eeg_channels(self) -> list[str]:
        return [name for name in self.channels if name.startswith(_EEG_PREFIX)]

    @property
    def other_channels(self) -> list[str]:
        return [name for name in self.channels if not name.startswith(_EEG_PREFIX)]

    # -- axe temporel -------------------------------------------------------
    @property
    def time(self) -> np.ndarray:
        return self.data[self.time_column].to_numpy()

    @property
    def duration(self) -> float:
        """Duree totale en secondes (dernier instant + une periode)."""
        return float(self.n_samples / self.sampling_rate)

    @property
    def sampling_period(self) -> float:
        return 1.0 / self.sampling_rate

    # -- acces aux donnees --------------------------------------------------
    def channel(self, name: str) -> np.ndarray:
        if name not in self.data.columns:
            raise KeyError(f"Canal inconnu : {name}. Disponibles : {self.channels}")
        return self.data[name].to_numpy()

    def window(self, start: float | None = None, end: float | None = None) -> pd.DataFrame:
        """Extrait la fenetre temporelle [start, end[ en secondes."""
        time = self.data[self.time_column]

        mask = pd.Series(True, index=self.data.index)
        if start is not None:
            mask &= time >= start
        if end is not None:
            mask &= time < end

        return self.data.loc[mask]

    def n_missing(self) -> int:
        """Nombre total de valeurs manquantes."""
        return int(self.data.isna().sum().sum())

    def __repr__(self) -> str:  # pragma: no cover - confort d'affichage
        return (
            f"Recording({self.recording_id}, {self.n_samples} echantillons, "
            f"{len(self.channels)} canaux, {self.sampling_rate:.0f} Hz, "
            f"{self.duration:.0f} s)"
        )


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
def load_events(path: Path | str) -> pd.DataFrame:
    """Lit les marqueurs du fichier `*h.mat` (Debut, YO, YF, HPN, SLI, ...).

    Retourne un DataFrame vide si le fichier ne suit pas la structure attendue :
    ces marqueurs sont un complement, ils ne sont pas indispensables au TP.
    """
    columns = ["onset_s", "label", "duration_s"]
    path = Path(path)

    try:
        pool = _read_object_pool(path)
        table = _find_timetable(pool)

        n_events = int(np.asarray(table["numRows"]).item())
        onsets_ms = _find_row_times_ms(pool, n_events)
        labels = _find_string_array(pool, n_events)

        if labels is None:
            labels = [f"event_{index + 1}" for index in range(n_events)]

        events = pd.DataFrame(
            {
                "onset_s": onsets_ms / _MS_PER_SECOND,
                "label": labels,
                "duration_s": np.nan,
            },
            columns=columns,
        )

        return events.sort_values("onset_s").reset_index(drop=True)

    except (ValueError, KeyError, OSError):
        return pd.DataFrame(columns=columns)


def load_recording(
    recording_id: str,
    root: Path | str | None = None,
    with_events: bool = True,
) -> Recording:
    """Charge un enregistrement complet a partir de son identifiant.

    Exemple
    -------
    >>> recording = load_recording("230102B_F_0000")
    >>> recording.sampling_rate
    256.0
    """
    folder = raw_dir(root)

    data_path = folder / f"{recording_id}d.mat"
    header_path = folder / f"{recording_id}h.mat"

    if not data_path.is_file():
        raise FileNotFoundError(f"Fichier de signal introuvable : {data_path}")

    pool = _read_object_pool(data_path)
    table = _find_timetable(pool)

    signal, n_blocks, samples_per_block = _stack_channels(table)
    row_times_s = _find_row_times_ms(pool, n_blocks) / _MS_PER_SECOND

    sampling_rate = _sampling_rate(row_times_s, samples_per_block)

    time = (
        row_times_s[:, None] + np.arange(samples_per_block)[None, :] / sampling_rate
    ).ravel()

    frame = pd.DataFrame(signal, columns=_channel_names(table))
    frame.insert(0, "Time", time)

    events = (
        load_events(header_path)
        if with_events and header_path.is_file()
        else pd.DataFrame(columns=["onset_s", "label", "duration_s"])
    )

    return Recording(
        recording_id=recording_id,
        data=frame,
        sampling_rate=sampling_rate,
        events=events,
        # Le champ `varUnits` du timetable est vide : l'unite n'est pas
        # enregistree dans le fichier. Le microvolt est la convention en EEG.
        amplitude_unit=None,
        source={
            "data_file": data_path.name,
            "header_file": header_path.name if header_path.is_file() else None,
            "blocks": n_blocks,
            "samples_per_block": samples_per_block,
        },
    )


# ---------------------------------------------------------------------------
# Annotations cliniques (fichier Excel) - utilisees a partir de la Partie II
# ---------------------------------------------------------------------------
def annotation_time_seconds(value, time_unit: str) -> float:
    """Convert an annotation using its explicitly supplied interpretation.

    ``minutes_seconds_centiseconds`` interprets displayed ``MM:SS:CC``.
    Excel stores these three groups as an HH:MM:SS day fraction, so recover
    the groups before applying their explicitly selected meaning. For example,
    Excel's 11:19:13 represents 11*60 + 19 + 13/100 = 679.13 seconds.
    """
    if time_unit == "minutes_seconds_centiseconds":
        if isinstance(value, str):
            parts = value.strip().split(":")
            if len(parts) != 3 or not all(part.isdigit() for part in parts):
                raise ValueError(f"Expected MM:SS:CC, got {value!r}")
            minutes, seconds, centiseconds = map(int, parts)
        else:
            value = float(value)
            if not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError("Expected an Excel time fraction in [0, 1).")
            packed_seconds = value * 86400
            rounded = round(packed_seconds)
            if not np.isclose(packed_seconds, rounded, rtol=0, atol=1e-5):
                raise ValueError("Excel time has sub-second groups; expected MM:SS:CC.")
            minutes, remainder = divmod(rounded, 3600)
            seconds, centiseconds = divmod(remainder, 60)
        if not 0 <= seconds < 60 or not 0 <= centiseconds < 100:
            raise ValueError(f"Invalid MM:SS:CC groups in {value!r}")
        return round(minutes * 60 + seconds + centiseconds / 100, 9)
    factors = {"minutes": 60.0, "seconds": 1.0, "excel_days": 86400.0}
    if time_unit not in factors:
        raise ValueError("Unknown annotation time unit.")
    return round(float(value) * factors[time_unit], 9)


def load_seizure_annotations(
    root: Path | str | None = None,
    filename: str = "Crises absences.xlsx",
    *,
    time_unit: str,
) -> pd.DataFrame:
    """Read one row per seizure, converting explicitly declared units to seconds.

    Use ``minutes_seconds_centiseconds`` for the legacy MM:SS:CC
    display convention. Other explicitly supported units are numeric minutes,
    seconds and Excel days. No time offset is inferred. Source values are
    retained for auditing the conversion.
    """
    units = ("minutes_seconds_centiseconds", "minutes", "seconds", "excel_days")
    if time_unit not in units:
        raise ValueError(f"time_unit must be one of {units}")
    table = pd.read_excel(raw_dir(root) / filename)
    table = table.rename(columns={table.columns[0]: "recording_id"})
    table.columns = [str(column).strip() for column in table.columns]
    table = table.rename(columns={"DebutCE5": "DebCE5", "DebtCE11": "DebCE11"})
    table = table.dropna(how="all")
    table = table[table["recording_id"].notna()].copy()
    numbers = sorted({
        int(column[len(prefix):])
        for column in table.columns
        for prefix in ("DebCE", "FinCE")
        if column.startswith(prefix) and column[len(prefix):].isdigit()
    })
    rows = []
    for _, record in table.iterrows():
        for number in numbers:
            start = record.get(f"DebCE{number}")
            end = record.get(f"FinCE{number}")
            if pd.isna(start) and pd.isna(end):
                continue
            if pd.isna(start) or pd.isna(end):
                raise ValueError(f"Incomplete seizure {number}: {record['recording_id']}")
            start_s = annotation_time_seconds(start, time_unit)
            end_s = annotation_time_seconds(end, time_unit)
            if not np.isfinite([start_s, end_s]).all() or start_s < 0 or end_s <= start_s:
                raise ValueError(f"Invalid seizure {number}: {record['recording_id']}")
            rows.append({
                "recording_id": str(record["recording_id"]).strip().upper(),
                "seizure": number,
                "start_value": start,
                "end_value": end,
                "annotation_unit": time_unit,
                "start_s": start_s,
                "end_s": end_s,
                "reported_count": record.get("NombreCE", np.nan),
            })
    seizures = pd.DataFrame(rows, columns=[
        "recording_id", "seizure", "start_value", "end_value", "annotation_unit",
        "start_s", "end_s", "reported_count",
    ])
    seizures["duration_s"] = seizures["end_s"] - seizures["start_s"]
    return seizures.sort_values(["recording_id", "start_s"]).reset_index(drop=True)


def match_recording_id(recording_id: str) -> str:
    """Convertit `230102B_F_0000` en `230102B-F` (forme utilisee dans l'Excel)."""
    base = recording_id.rsplit("_", 1)[0]
    return base.replace("_", "-").upper()


# ---------------------------------------------------------------------------
# Resumes pour la Question 1
# ---------------------------------------------------------------------------
def _format_duration(seconds: float) -> str:
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes):d} min {remainder:04.1f} s"


def summarize(recording: Recording) -> pd.DataFrame:
    """Tableau de synthese demande a la Question 1."""
    data = recording.data
    numeric = data[recording.channels]

    rows = [
        ("Fichier de signal", recording.source.get("data_file", "-")),
        ("Fichier d'entete", recording.source.get("header_file", "-")),
        ("Nombre de lignes (echantillons)", f"{recording.n_samples:,}".replace(",", " ")),
        ("Nombre de variables (colonnes)", recording.n_variables),
        ("dont canaux de mesure", len(recording.channels)),
        ("dont canaux EEG", len(recording.eeg_channels)),
        ("Canaux non EEG", ", ".join(recording.other_channels) or "aucun"),
        ("Frequence d'echantillonnage", f"{recording.sampling_rate:.1f} Hz"),
        ("Periode d'echantillonnage", f"{recording.sampling_period * 1000:.3f} ms"),
        ("Duree totale", f"{recording.duration:.1f} s ({_format_duration(recording.duration)})"),
        ("Premier instant", f"{recording.time[0]:.3f} s"),
        ("Dernier instant", f"{recording.time[-1]:.3f} s"),
        ("Valeurs manquantes (NaN)", int(numeric.isna().sum().sum())),
        ("Valeurs infinies", int(np.isinf(numeric.to_numpy()).sum())),
        ("Unite de temps", "seconde"),
        (
            "Unite d'amplitude",
            recording.amplitude_unit
            or "non renseignee dans le fichier",
        ),
        ("Marqueurs dans l'entete", len(recording.events)),
        ("Taille en memoire", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} Mo"),
    ]

    return pd.DataFrame(rows, columns=["Propriete", "Valeur"])


def channel_quality(recording: Recording) -> pd.DataFrame:
    """Statistiques par canal : valeurs manquantes et ordre de grandeur."""
    rows = []

    for name in recording.channels:
        values = recording.channel(name)

        rows.append(
            {
                "canal": name,
                "type": "EEG" if name.startswith(_EEG_PREFIX) else "autre",
                "manquantes": int(np.isnan(values).sum()),
                "infinies": int(np.isinf(values).sum()),
                "min": values.min(),
                "max": values.max(),
                "moyenne": values.mean(),
                "ecart_type": values.std(),
            }
        )

    return pd.DataFrame(rows).round(2)


def missing_by_column(recording: Recording) -> pd.DataFrame:
    """Detail des valeurs manquantes, colonne par colonne."""
    missing = recording.data.isna().sum()

    return pd.DataFrame(
        {
            "colonne": missing.index,
            "manquantes": missing.to_numpy(),
            "pourcentage": (missing.to_numpy() / len(recording.data) * 100).round(3),
        }
    )


def saturation_report(recording: Recording, tolerance: float = 0.999) -> pd.DataFrame:
    """Repere les echantillons satures et les canaux constants.

    Un canal sature quand l'amplitude depasse la plage du convertisseur : les
    valeurs restent collees a une butee, identique pour tous les canaux. Ces
    echantillons ne sont pas manquants au sens de pandas, mais ils ne portent
    aucune information utile.
    """
    signal = recording.data[recording.channels].to_numpy()
    rail = float(np.nanmax(np.abs(signal)))

    rows = []
    for name in recording.channels:
        values = recording.channel(name)
        spread = float(values.std())

        if spread == 0.0:
            hits, remark = 0, "canal constant"
        else:
            hits = int((np.abs(values) >= tolerance * rail).sum())
            remark = "saturation" if hits > 1 else ""

        rows.append(
            {
                "canal": name,
                "maximum_absolu": round(float(np.abs(values).max()), 1),
                "echantillons_satures": hits,
                "pourcentage": round(hits / values.size * 100, 3),
                "ecart_type": round(spread, 1),
                "remarque": remark,
            }
        )

    report = pd.DataFrame(rows)
    report = report.sort_values("echantillons_satures", ascending=False)

    return report.reset_index(drop=True)


def align_clock_annotations(annotations: pd.DataFrame, recording_start: str) -> pd.DataFrame:
    """Convert Excel clock seconds to elapsed seconds using a supplied start clock.

    A midnight rollover is supported for recordings lasting less than 24 hours.
    The start clock must come from the recording, never from signal appearance.
    """
    from datetime import datetime
    clock = datetime.strptime(recording_start, '%H:%M:%S')
    origin = clock.hour * 3600 + clock.minute * 60 + clock.second
    if not annotations.annotation_unit.eq('excel_days').all():
        raise ValueError('Load clock annotations with time_unit="excel_days".')
    result = annotations.copy()
    duration = result.end_s - result.start_s
    result['start_s'] = (result.start_s - origin) % 86400
    result['end_s'] = result.start_s + duration
    result['duration_s'] = duration
    result['recording_start_clock'] = recording_start
    return result.sort_values('start_s').reset_index(drop=True)
