"""Interface Streamlit du mini-projet EEG : exploration et pretraitement.

Lancement (depuis la racine du projet) :

    streamlit run app.py

L'interface ne contient aucune logique de traitement ni de trace : elle appelle
les memes fonctions que `notebooks/main.ipynb` (`src.data`, `src.preprocessing`,
`src.visualization`), pour que les figures vues ici soient exactement celles du
rapport.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (
    Recording,
    channel_quality,
    list_recordings,
    load_recording,
    load_seizure_annotations,
    match_recording_id,
    missing_by_column,
    saturation_report,
    summarize,
)
from src.preprocessing import DEFAULT_BAND, band_powers, clean_recording, mains_frequency
from src.visualization import (
    plot_band_powers,
    plot_montage,
    plot_overview,
    plot_signal,
    plot_spectrum,
)

st.set_page_config(page_title="EEG - crises d'absence", page_icon="🧠", layout="wide")

# Longueurs de fenetre proposees. Le montage empile trace tous les points sans
# enveloppe : au-dela de 30 s la figure devient lente pour rien.
WINDOW_LENGTHS = [2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
MONTAGE_LENGTHS = [2.0, 5.0, 10.0, 30.0]
BAND_NAMES = ("delta", "theta", "alpha", "beta")


# ---------------------------------------------------------------------------
# Chargement (mis en cache : relire un .mat prend plusieurs secondes)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_inventory() -> pd.DataFrame:
    return list_recordings(ROOT)


@st.cache_resource(show_spinner=False, max_entries=3)
def get_recording(recording_id: str) -> Recording:
    return load_recording(recording_id, root=ROOT)


@st.cache_resource(show_spinner=False, max_entries=3)
def get_cleaned(
    recording_id: str,
    band: tuple[float, float],
    notch: float | None,
) -> Recording:
    return clean_recording(get_recording(recording_id), band=band, notch=notch)


@st.cache_data(show_spinner=False)
def get_mains(recording_id: str) -> float:
    recording = get_recording(recording_id)
    return mains_frequency(
        recording.channel(recording.eeg_channels[0]),
        recording.sampling_rate,
    )


@st.cache_data(show_spinner=False)
def get_annotations() -> pd.DataFrame:
    try:
        return load_seizure_annotations(ROOT)
    except FileNotFoundError:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Aides a l'affichage
# ---------------------------------------------------------------------------
def show(fig: plt.Figure, filename: str | None = None) -> None:
    """Affiche une figure, propose son export en PNG, puis la ferme."""
    fig.tight_layout()
    st.pyplot(fig, width="stretch")

    # Encoder un PNG coute ~0,3 s par figure, sur les huit figures et a chaque
    # rafraichissement : on ne le fait que si l'export est demande.
    if filename is not None and st.session_state.get("exports", False):
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        st.download_button(
            "Telecharger la figure (PNG)",
            buffer.getvalue(),
            file_name=filename,
            mime="image/png",
            key=f"download_{filename}",
        )

    plt.close(fig)


def table(frame: pd.DataFrame, **kwargs) -> None:
    st.dataframe(frame, hide_index=True, width="stretch", **kwargs)


def _goto_event(key: str, duration: float, onsets: dict[str, float]) -> None:
    """Cale la fenetre sur un marqueur avant que le curseur ne soit redessine."""
    choice = st.session_state.get(f"{key}_goto")
    if choice not in onsets:
        return

    length = float(st.session_state.get(f"{key}_len", 10.0))
    max_start = max(duration - length, 0.0)

    # Le marqueur est place au premier quart : on voit ce qui le precede
    # autant que ce qui le suit.
    target = max(0.0, min(onsets[choice] - length / 4.0, max_start))
    st.session_state[f"{key}_start"] = round(target * 2.0) / 2.0


def window_controls(
    recording: Recording,
    key: str,
    lengths: list[float],
    default_length: float = 10.0,
) -> tuple[float, float]:
    """Curseur de position + duree, avec saut direct vers un marqueur."""
    duration = recording.duration
    events = recording.events

    columns = st.columns([1, 3, 2] if not events.empty else [1, 5])

    length = float(
        columns[0].selectbox(
            "Duree (s)",
            lengths,
            index=lengths.index(default_length) if default_length in lengths else 0,
            key=f"{key}_len",
        )
    )
    max_start = max(duration - length, 0.0)

    if not events.empty:
        onsets = {
            f"{row.onset_s:.1f} s - {row.label}": float(row.onset_s)
            for row in events.itertuples()
        }
        columns[2].selectbox(
            "Aller a un marqueur",
            ["-", *onsets],
            key=f"{key}_goto",
            on_change=_goto_event,
            args=(key, duration, onsets),
        )

    if max_start <= 0.0:
        columns[1].caption("Enregistrement plus court que la fenetre : vue complete.")
        return 0.0, duration

    # La duree a pu diminuer la plage admissible depuis le dernier passage.
    st.session_state.setdefault(f"{key}_start", 0.0)
    st.session_state[f"{key}_start"] = min(
        float(st.session_state[f"{key}_start"]), max_start
    )

    start = float(
        columns[1].slider(
            "Debut de la fenetre (s)",
            min_value=0.0,
            max_value=float(round(max_start, 1)),
            step=0.5 if length <= 10.0 else 1.0,
            key=f"{key}_start",
        )
    )

    return start, start + length


# ---------------------------------------------------------------------------
# Barre laterale : enregistrement et pretraitement
# ---------------------------------------------------------------------------
st.sidebar.title("EEG - crises d'absence")

inventory = get_inventory()
if inventory.empty:
    st.error("Aucun fichier `*d.mat` trouve dans `data/raw`.")
    st.stop()

recording_id = st.sidebar.selectbox(
    "Enregistrement",
    inventory["recording_id"].tolist(),
    help="Un couple de fichiers `<id>d.mat` (signal) et `<id>h.mat` (marqueurs).",
)

with st.spinner(f"Lecture de {recording_id}..."):
    raw = get_recording(recording_id)

detected = get_mains(recording_id)
nyquist = raw.sampling_rate / 2.0

st.sidebar.divider()
st.sidebar.subheader("Pretraitement")
st.sidebar.caption(
    f"Secteur detecte sur {raw.eeg_channels[0]} : **{detected:.0f} Hz** "
    "(pic de densite spectrale le plus fort entre 50 et 60 Hz)."
)

apply_filter = st.sidebar.toggle("Appliquer le filtrage", value=True)

notch_options: dict[str, float | None] = {
    f"Automatique - {detected:.0f} Hz": detected,
    "50 Hz": 50.0,
    "60 Hz": 60.0,
    "Aucun": None,
}
notch = notch_options[
    st.sidebar.radio("Coupe-bande secteur", list(notch_options), disabled=not apply_filter)
]

band = st.sidebar.slider(
    "Passe-bande (Hz)",
    min_value=0.5,
    max_value=float(round(nyquist - 1.0)),
    value=DEFAULT_BAND,
    step=0.5,
    disabled=not apply_filter,
)

if apply_filter:
    with st.spinner("Filtrage du signal..."):
        recording = get_cleaned(recording_id, tuple(band), notch)
    notch_state = f"coupe-bande {notch:.0f} Hz" if notch else "sans coupe-bande"
    state = f"filtre - {notch_state}, passe-bande {band[0]:g}-{band[1]:g} Hz"
else:
    recording = raw
    state = "brut (aucun filtrage)"

st.sidebar.divider()
st.sidebar.checkbox(
    "Proposer le telechargement des figures",
    value=False,
    key="exports",
    help="Genere un PNG sous chaque figure. A activer au moment d'exporter pour le rapport.",
)

channel = st.sidebar.selectbox(
    "Canal principal",
    raw.channels,
    index=raw.channels.index(raw.eeg_channels[0]),
    help="Utilise par les onglets Signal, Spectre et Bandes.",
)

# ---------------------------------------------------------------------------
# En-tete
# ---------------------------------------------------------------------------
st.title(f"Enregistrement {recording_id}")
st.caption(f"Signal affiche : {state}")

metrics = st.columns(5)
metrics[0].metric("Echantillons", f"{raw.n_samples:,}".replace(",", " "))
metrics[1].metric("Canaux EEG", f"{len(raw.eeg_channels)} / {len(raw.channels)}")
metrics[2].metric("Frequence", f"{raw.sampling_rate:.0f} Hz")
metrics[3].metric("Duree", f"{raw.duration / 60:.1f} min")
metrics[4].metric("Valeurs manquantes", f"{raw.n_missing()}")

(
    tab_apercu,
    tab_signal,
    tab_montage,
    tab_spectre,
    tab_bandes,
    tab_qualite,
    tab_marqueurs,
) = st.tabs(
    ["Apercu", "Signal", "Montage", "Spectre", "Bandes", "Qualite", "Marqueurs & crises"]
)

# ---------------------------------------------------------------------------
with tab_apercu:
    st.subheader("Vue d'ensemble")
    st.caption(
        "Axe vertical cadre sur 99,5 % des valeurs : les artefacts de quelques "
        "milliers de microvolts n'ecrasent plus le trace. Les traits orange sont "
        "les marqueurs du manipulateur."
    )

    fig, ax = plt.subplots(figsize=(13, 3.4))
    plot_overview(recording, channel=channel, ax=ax)
    show(fig, f"{recording_id}_apercu_{channel}.png")

    left, right = st.columns([2, 3])
    left.markdown("**Synthese du fichier**")
    with left:
        table(summarize(raw).astype({"Valeur": str}))
    right.markdown("**Inventaire des enregistrements**")
    with right:
        table(inventory)

# ---------------------------------------------------------------------------
with tab_signal:
    st.subheader(f"Canal {channel}")
    start, end = window_controls(recording, "signal", WINDOW_LENGTHS)

    options = st.columns(2)
    compare = options[0].checkbox("Comparer brut / filtre", value=True, disabled=not apply_filter)
    show_events = options[1].checkbox("Afficher les marqueurs", value=True, key="sig_events")

    if compare and apply_filter:
        # Deux amplitudes tres differentes : deux axes separes, jamais deux
        # echelles superposees sur la meme figure.
        fig, (ax_raw, ax_clean) = plt.subplots(2, 1, figsize=(13, 5.6), sharex=True)
        plot_signal(
            raw,
            channel=channel,
            start=start,
            end=end,
            ax=ax_raw,
            ylim="robust",
            title=f"Brut - {channel} ({start:.0f}-{end:.0f} s)",
        )
        plot_signal(
            recording,
            channel=channel,
            start=start,
            end=end,
            ax=ax_clean,
            ylim="robust",
            show_events=show_events,
            title=f"Filtre - {channel} ({start:.0f}-{end:.0f} s)",
        )
        ax_raw.set_xlabel("")
    else:
        fig, ax = plt.subplots(figsize=(13, 3.4))
        plot_signal(
            recording,
            channel=channel,
            start=start,
            end=end,
            ax=ax,
            ylim="robust",
            show_events=show_events,
        )

    show(fig, f"{recording_id}_signal_{channel}.png")

# ---------------------------------------------------------------------------
with tab_montage:
    st.subheader("Montage multi-canaux")
    st.caption(
        "Chaque canal est centre sur sa moyenne dans la fenetre puis decale "
        "verticalement d'un pas constant, comme sur un trace clinique."
    )

    selected = st.multiselect("Canaux", raw.channels, default=raw.eeg_channels)
    start, end = window_controls(recording, "montage", MONTAGE_LENGTHS)
    montage_events = st.checkbox("Afficher les marqueurs", value=False, key="mont_events")

    if not selected:
        st.info("Selectionnez au moins un canal.")
    else:
        fig, ax = plt.subplots(figsize=(13, 0.42 * len(selected) + 1.8))
        plot_montage(
            recording,
            channels=selected,
            start=start,
            end=end,
            ax=ax,
            show_events=montage_events,
        )
        show(fig, f"{recording_id}_montage.png")

# ---------------------------------------------------------------------------
with tab_spectre:
    st.subheader(f"Densite spectrale - canal {channel}")
    st.caption(
        "Methode de Welch. C'est la figure qui justifie le pretraitement : la "
        "raie du secteur y est visible d'un coup d'oeil."
    )

    start, end = window_controls(recording, "spectre", WINDOW_LENGTHS, default_length=60.0)
    f_max = st.slider(
        "Frequence maximale affichee (Hz)",
        min_value=20.0,
        max_value=float(round(nyquist)),
        value=min(80.0, float(round(nyquist))),
        step=5.0,
    )

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        plot_spectrum(
            raw,
            channel=channel,
            start=start,
            end=end,
            f_max=f_max,
            ax=ax,
            highlight=detected,
            title=f"Brut - {channel}",
        )
        show(fig, f"{recording_id}_spectre_brut.png")

    with right:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        plot_spectrum(
            recording,
            channel=channel,
            start=start,
            end=end,
            f_max=f_max,
            ax=ax,
            highlight=detected,
            title=f"{'Filtre' if apply_filter else 'Brut'} - {channel}",
        )
        show(fig, f"{recording_id}_spectre_filtre.png")

# ---------------------------------------------------------------------------
with tab_bandes:
    st.subheader("Puissance par bande de frequence")
    st.caption(
        "delta 0,5-4 Hz, theta 4-8 Hz, alpha 8-13 Hz, beta 13-30 Hz. Ces "
        "puissances serviront de caracteristiques a partir de la Partie VI."
    )

    start, end = window_controls(recording, "bandes", WINDOW_LENGTHS)
    window = recording.window(start, end)

    if window.empty:
        st.warning("Fenetre vide.")
    else:
        powers = band_powers(window[channel].to_numpy(), recording.sampling_rate)
        total = float(powers.sum()) or 1.0

        left, right = st.columns([3, 2])

        with left:
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            plot_band_powers(powers, ax=ax, title=f"{channel} - {start:.0f} a {end:.0f} s")
            show(fig, f"{recording_id}_bandes_{channel}.png")

        with right:
            st.markdown("**Valeurs**")
            table(
                pd.DataFrame(
                    {
                        "bande": powers.index,
                        "puissance (uV2)": powers.to_numpy().round(2),
                        "part (%)": (powers / total * 100).to_numpy().round(1),
                    }
                )
            )

        st.markdown("**Repartition sur tous les canaux EEG**")
        rows = []
        for name in recording.eeg_channels:
            shares = band_powers(window[name].to_numpy(), recording.sampling_rate)
            shares = shares / (float(shares.sum()) or 1.0) * 100.0
            rows.append({"canal": name, **shares.round(1).to_dict()})

        table(
            pd.DataFrame(rows),
            column_config={
                name: st.column_config.ProgressColumn(
                    name, format="%.1f %%", min_value=0.0, max_value=100.0
                )
                for name in BAND_NAMES
            },
        )

# ---------------------------------------------------------------------------
with tab_qualite:
    st.subheader("Qualite des donnees")
    st.caption("Calcule sur le signal brut : c'est lui qui porte les defauts.")

    st.markdown("**Statistiques par canal**")
    table(channel_quality(raw))

    left, right = st.columns(2)
    left.markdown("**Valeurs manquantes par colonne**")
    with left:
        table(missing_by_column(raw))

    right.markdown("**Saturations et canaux constants**")
    with right:
        table(saturation_report(raw))

# ---------------------------------------------------------------------------
with tab_marqueurs:
    st.subheader("Marqueurs du manipulateur")

    if raw.events.empty:
        st.info("Aucun marqueur lisible dans le fichier d'en-tete.")
    else:
        table(raw.events)

    st.subheader("Annotations de crises")
    annotations = get_annotations()

    if annotations.empty:
        st.info("Fichier `Annotations_MAJ.xlsx` absent de `data/raw`.")
    else:
        st.caption(
            "Source : `Annotations_MAJ.xlsx`. `start_s` et `end_s` sont en secondes "
            "depuis le debut du trace (heure de la crise - `DebutTrace`), donc sur "
            "le meme axe que les figures."
        )

        key = match_recording_id(recording_id)
        matched = annotations[annotations["recording_id"].str.upper() == key]

        if matched.empty:
            st.info(f"Aucune crise annotee pour `{key}`.")
        else:
            table(matched)
