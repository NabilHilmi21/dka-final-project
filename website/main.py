from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from models import (
    RISK_OUTPUT,
    bulan_akhir,
    bulan_awal,
    bulan_tengah,
    contoh_rules,
    freq_kecamatan_rendah,
    freq_kecamatan_sedang,
    freq_kecamatan_tinggi,
    freq_profesi_rendah,
    freq_profesi_sedang,
    freq_profesi_tinggi,
    fuzzification,
    inferensi,
    jam_normal,
    jam_sepi,
    jam_sibuk,
    jumlah_rules,
    karakteristik_rendah,
    karakteristik_sedang,
    karakteristik_tinggi,
    kategori_risiko,
    mamdani,
    risiko_rendah,
    risiko_sedang,
    risiko_tinggi,
    sugeno,
)
from schemas import prepare_data


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "datakecminim.csv"
KATEGORI_ORDER = ["RENDAH", "SEDANG", "TINGGI"]
SKOR_AKTUAL = {
    "RENDAH": 25,
    "SEDANG": 50,
    "TINGGI": 85,
}
INPUT_FUZZY = [
    "JAM",
    "BULAN",
    "FREQ_KECAMATAN",
    "FREQ_PROFESI",
    "SKOR_KARAKTERISTIK",
]
OUTPUT_COLUMNS = [
    "JAM_ASLI",
    "BULAN",
    "KECAMATAN",
    "FREQ_KECAMATAN",
    "PROFESI",
    "FREQ_PROFESI",
    "KARAKTERISTIK LAKA",
    "SKOR_KARAKTERISTIK",
    "KODE_CIDERA",
    "KATEGORI_AKTUAL",
    "NILAI_MAMDANI",
    "KATEGORI_MAMDANI",
    "NILAI_SUGENO",
    "KATEGORI_SUGENO",
]


st.set_page_config(
    page_title="Fuzzy Logic Risiko Kecelakaan",
    layout="wide",
)

@st.cache_data(show_spinner=False)
def load_raw_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_clean_data():
    return prepare_data(DATA_PATH)


def hitung_risiko(row):
    nilai_mamdani = mamdani(
        row["JAM"],
        row["BULAN"],
        row["FREQ_KECAMATAN"],
        row["FREQ_PROFESI"],
        row["SKOR_KARAKTERISTIK"],
    )
    nilai_sugeno = sugeno(
        row["JAM"],
        row["BULAN"],
        row["FREQ_KECAMATAN"],
        row["FREQ_PROFESI"],
        row["SKOR_KARAKTERISTIK"],
    )

    return pd.Series(
        {
            "NILAI_MAMDANI": round(nilai_mamdani, 2),
            "KATEGORI_MAMDANI": kategori_risiko(nilai_mamdani),
            "NILAI_SUGENO": round(nilai_sugeno, 2),
            "KATEGORI_SUGENO": kategori_risiko(nilai_sugeno),
        }
    )


@st.cache_data(show_spinner=True)
def get_result_data():
    df = load_clean_data()
    hasil_fuzzy = df.apply(hitung_risiko, axis=1)
    return pd.concat([df, hasil_fuzzy], axis=1)


@st.cache_data(show_spinner=False)
def get_evaluation(df):
    nilai_aktual = df["KATEGORI_AKTUAL"].map(SKOR_AKTUAL)

    benar_mamdani = (df["KATEGORI_MAMDANI"] == df["KATEGORI_AKTUAL"]).sum()
    benar_sugeno = (df["KATEGORI_SUGENO"] == df["KATEGORI_AKTUAL"]).sum()
    total = len(df)

    mae_mamdani = (df["NILAI_MAMDANI"] - nilai_aktual).abs().mean()
    mae_sugeno = (df["NILAI_SUGENO"] - nilai_aktual).abs().mean()

    return pd.DataFrame(
        {
            "Metode": ["Mamdani", "Sugeno"],
            "Benar": [benar_mamdani, benar_sugeno],
            "Total": [total, total],
            "Akurasi (%)": [
                benar_mamdani / total * 100 if total else 0,
                benar_sugeno / total * 100 if total else 0,
            ],
            "MAE": [mae_mamdani, mae_sugeno],
        }
    )


@st.cache_data(show_spinner=False)
def get_mae_scores(df):
    nilai_aktual = df["KATEGORI_AKTUAL"].map(SKOR_AKTUAL)
    mae_df = df[["KATEGORI_AKTUAL"]].copy()
    mae_df["Mamdani"] = (df["NILAI_MAMDANI"] - nilai_aktual).abs()
    mae_df["Sugeno"] = (df["NILAI_SUGENO"] - nilai_aktual).abs()
    return mae_df


def intro():
    st.title("Fuzzy Logic Risiko Kecelakaan")
    st.caption("Website ini mengubah alur notebook menjadi tampilan interaktif.")

    st.markdown(
        """
        Sistem ini memprediksi kategori risiko kecelakaan dengan dua metode fuzzy:
        **Mamdani** dan **Sugeno**. Semua fungsi keanggotaan, rule base,
        inferensi, defuzzifikasi, evaluasi, dan visualisasi mengikuti kode
        yang ada di notebook `fuzzy_kecelakaan.ipynb`.
        """
    )


def metric_cards(df_raw, df_clean, df_result=None):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw rows", f"{len(df_raw):,}")
    col2.metric("Clean rows", f"{len(df_clean):,}")
    col3.metric("Input fuzzy", len(INPUT_FUZZY))
    col4.metric("Rule base", jumlah_rules())

    if df_result is not None:
        evaluasi = get_evaluation(df_result)
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Akurasi Mamdani", f"{evaluasi.loc[0, 'Akurasi (%)']:.2f}%")
        col_b.metric("Akurasi Sugeno", f"{evaluasi.loc[1, 'Akurasi (%)']:.2f}%")
        col_c.metric("MAE Mamdani", f"{evaluasi.loc[0, 'MAE']:.2f}")
        col_d.metric("MAE Sugeno", f"{evaluasi.loc[1, 'MAE']:.2f}")


def section_dataset(df_raw, df_clean):
    st.header("1. Membaca Dataset")
    st.write("Sumber dataset: Kaggle - data kecelakaan.")
    metric_cards(df_raw, df_clean)

    st.subheader("Kolom Raw")
    st.write(list(df_raw.columns))

    st.subheader("Preview Raw Data")
    st.dataframe(df_raw.head(20), use_container_width=True)

    st.subheader("Ringkasan Missing Value")
    missing = df_raw.isna().sum().reset_index()
    missing.columns = ["Kolom", "Jumlah missing"]
    st.dataframe(missing, use_container_width=True)


def section_preprocessing(df_raw, df_clean):
    st.header("2. Preprocessing dan Data Cleaning")
    st.markdown(
        """
        Cleaning yang dilakukan:

        - Mengambil kolom penting untuk sistem fuzzy.
        - Membersihkan teks menjadi huruf besar dan menghapus spasi berlebih.
        - Mengganti nilai kosong atau `-` menjadi `TIDAK DIKETAHUI`.
        - Mengubah jam ke angka 0 sampai 23.99.
        - Membatasi bulan ke rentang 1 sampai 12.
        - Membuat fitur `FREQ_KECAMATAN`, `FREQ_PROFESI`, dan `SKOR_KARAKTERISTIK`.
        - Membuat label evaluasi `KATEGORI_AKTUAL` dari `KODE_CIDERA`.
        """
    )

    st.subheader("Data Setelah Cleaning")
    st.dataframe(df_clean.head(30), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Distribusi Kategori Aktual")
        st.dataframe(
            df_clean["KATEGORI_AKTUAL"]
            .value_counts()
            .reindex(KATEGORI_ORDER, fill_value=0),
            use_container_width=True,
        )
    with col2:
        st.write("Top Kecamatan")
        st.dataframe(df_clean["KECAMATAN"].value_counts().head(10), use_container_width=True)
    with col3:
        st.write("Top Profesi")
        st.dataframe(df_clean["PROFESI"].value_counts().head(10), use_container_width=True)

    st.subheader("Kolom yang Dipakai")
    st.dataframe(
        pd.DataFrame(
            {
                "Input fuzzy": INPUT_FUZZY,
                "Keterangan": [
                    "Jam kejadian setelah parsing",
                    "Bulan kejadian",
                    "Frekuensi kecamatan di dataset",
                    "Frekuensi profesi di dataset",
                    "Skor dari karakteristik laka",
                ],
            }
        ),
        use_container_width=True,
    )


def plot_membership(title, x_values, series):
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, y_values in series.items():
        ax.plot(x_values, y_values, label=label, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_ylabel("Degree of membership")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def section_membership(df_clean):
    st.header("3. Variable Linguistic dan Fungsi Keanggotaan")
    st.write("Semua grafik di bawah dibuat dari fungsi yang dipakai model.")

    x_jam = list(range(0, 24))
    x_bulan = list(range(1, 13))
    max_kecamatan = int(max(5065, df_clean["FREQ_KECAMATAN"].max()))
    max_profesi = int(max(12000, df_clean["FREQ_PROFESI"].max()))
    x_kecamatan = list(range(0, max_kecamatan + 1, 50))
    x_profesi = list(range(0, max_profesi + 1, 100))
    x_karakteristik = list(range(0, 101))
    x_risiko = list(range(0, 101))

    fig1 = plot_membership(
        "JAM",
        x_jam,
        {
            "sepi": [jam_sepi(x) for x in x_jam],
            "normal": [jam_normal(x) for x in x_jam],
            "sibuk": [jam_sibuk(x) for x in x_jam],
        },
    )
    fig2 = plot_membership(
        "BULAN",
        x_bulan,
        {
            "awal": [bulan_awal(x) for x in x_bulan],
            "tengah": [bulan_tengah(x) for x in x_bulan],
            "akhir": [bulan_akhir(x) for x in x_bulan],
        },
    )
    fig3 = plot_membership(
        "FREQ_KECAMATAN",
        x_kecamatan,
        {
            "rendah": [freq_kecamatan_rendah(x) for x in x_kecamatan],
            "sedang": [freq_kecamatan_sedang(x) for x in x_kecamatan],
            "tinggi": [freq_kecamatan_tinggi(x) for x in x_kecamatan],
        },
    )
    fig4 = plot_membership(
        "FREQ_PROFESI",
        x_profesi,
        {
            "rendah": [freq_profesi_rendah(x) for x in x_profesi],
            "sedang": [freq_profesi_sedang(x) for x in x_profesi],
            "tinggi": [freq_profesi_tinggi(x) for x in x_profesi],
        },
    )
    fig5 = plot_membership(
        "SKOR_KARAKTERISTIK",
        x_karakteristik,
        {
            "rendah": [karakteristik_rendah(x) for x in x_karakteristik],
            "sedang": [karakteristik_sedang(x) for x in x_karakteristik],
            "tinggi": [karakteristik_tinggi(x) for x in x_karakteristik],
        },
    )
    fig6 = plot_membership(
        "RISIKO",
        x_risiko,
        {
            "rendah": [risiko_rendah(x) for x in x_risiko],
            "sedang": [risiko_sedang(x) for x in x_risiko],
            "tinggi": [risiko_tinggi(x) for x in x_risiko],
        },
    )

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
        st.pyplot(fig3)
        st.pyplot(fig5)
    with col2:
        st.pyplot(fig2)
        st.pyplot(fig4)
        st.pyplot(fig6)


def row_selector(df_clean):
    max_index = len(df_clean) - 1
    index = st.slider("Pilih index data contoh", 0, max_index, 0)
    return index, df_clean.iloc[index]


def section_fuzzification_rules(df_clean):
    st.header("4. Fuzzifikasi dan Rule Base")
    index, row = row_selector(df_clean)

    st.subheader(f"Data Contoh Index {index}")
    st.dataframe(row.to_frame("Nilai"), use_container_width=True)

    fuzzy = fuzzification(
        row["JAM"],
        row["BULAN"],
        row["FREQ_KECAMATAN"],
        row["FREQ_PROFESI"],
        row["SKOR_KARAKTERISTIK"],
    )
    st.subheader("Hasil Fuzzifikasi")
    fuzzy_rows = []
    for variable, values in fuzzy.items():
        for label, degree in values.items():
            fuzzy_rows.append(
                {
                    "Variabel": variable,
                    "Linguistic value": label,
                    "Degree": round(degree, 4),
                }
            )
    st.dataframe(pd.DataFrame(fuzzy_rows), use_container_width=True)

    st.subheader("Rule Base")
    st.write(f"Jumlah rule: **{jumlah_rules()}**")
    st.dataframe(pd.DataFrame(contoh_rules(15)), use_container_width=True)


def section_inference(df_clean):
    st.header("5. Inferensi dan Defuzzifikasi")
    index, row = row_selector(df_clean)

    fuzzy = fuzzification(
        row["JAM"],
        row["BULAN"],
        row["FREQ_KECAMATAN"],
        row["FREQ_PROFESI"],
        row["SKOR_KARAKTERISTIK"],
    )
    hasil_inferensi, aktif = inferensi(fuzzy)
    nilai_mamdani = mamdani(
        row["JAM"],
        row["BULAN"],
        row["FREQ_KECAMATAN"],
        row["FREQ_PROFESI"],
        row["SKOR_KARAKTERISTIK"],
    )
    nilai_sugeno = sugeno(
        row["JAM"],
        row["BULAN"],
        row["FREQ_KECAMATAN"],
        row["FREQ_PROFESI"],
        row["SKOR_KARAKTERISTIK"],
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Inferensi rendah", f"{hasil_inferensi['rendah']:.3f}")
    col2.metric("Inferensi sedang", f"{hasil_inferensi['sedang']:.3f}")
    col3.metric("Inferensi tinggi", f"{hasil_inferensi['tinggi']:.3f}")
    col4.metric("Rule aktif", len(aktif))

    col_a, col_b = st.columns(2)
    col_a.metric("Nilai Mamdani", f"{nilai_mamdani:.2f}", kategori_risiko(nilai_mamdani))
    col_b.metric("Nilai Sugeno", f"{nilai_sugeno:.2f}", kategori_risiko(nilai_sugeno))

    st.subheader("Rule Aktif")
    aktif_rows = []
    for rule_index, rule, strength in aktif[:30]:
        aktif_rows.append(
            {
                "No rule": rule_index,
                "jam": rule["jam"],
                "bulan": rule["bulan"],
                "freq_kecamatan": rule["freq_kecamatan"],
                "freq_profesi": rule["freq_profesi"],
                "karakteristik": rule["karakteristik"],
                "output": rule["output"],
                "strength": round(strength, 4),
            }
        )
    st.dataframe(pd.DataFrame(aktif_rows), use_container_width=True)

    st.info(
        "Mamdani memakai centroid pada domain 0-100. Sugeno memakai weighted average "
        f"dengan konstanta output {RISK_OUTPUT}."
    )


def section_results(df_result):
    st.header("6. Menghitung Semua Data")

    kategori_filter = st.multiselect(
        "Filter kategori aktual",
        KATEGORI_ORDER,
        default=KATEGORI_ORDER,
    )
    metode_filter = st.selectbox(
        "Urutkan berdasarkan",
        ["NILAI_MAMDANI", "NILAI_SUGENO", "JAM", "FREQ_KECAMATAN", "FREQ_PROFESI"],
    )
    filtered = df_result[df_result["KATEGORI_AKTUAL"].isin(kategori_filter)]
    filtered = filtered.sort_values(metode_filter, ascending=False)

    st.write(f"Menampilkan {len(filtered):,} data.")
    st.dataframe(filtered[OUTPUT_COLUMNS].head(200), use_container_width=True)

    csv = filtered[OUTPUT_COLUMNS].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download hasil filter sebagai CSV",
        data=csv,
        file_name="hasil_fuzzy_kecelakaan.csv",
        mime="text/csv",
    )


def bar_chart(title, labels, values, ylabel, colors=None, ylim=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}" if isinstance(value, float) else str(value),
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    return fig


def grouped_bar_chart(title, categories, series, ylabel, colors=None):
    x_positions = range(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))

    for index, (label, values) in enumerate(series.items()):
        offsets = [x + (index - 0.5) * width for x in x_positions]
        bars = ax.bar(
            offsets,
            values,
            width,
            label=label,
            color=colors[index] if colors else None,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(categories)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def section_evaluation(df_result):
    st.header("7. Evaluasi Akurasi, MAE, dan Confusion Matrix")
    evaluasi = get_evaluation(df_result)
    mae_scores = get_mae_scores(df_result)
    metric_cards(load_raw_data(), load_clean_data(), df_result)

    st.subheader("Tabel Evaluasi")
    st.dataframe(evaluasi, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(
            bar_chart(
                "Perbandingan Akurasi",
                evaluasi["Metode"],
                evaluasi["Akurasi (%)"],
                "Akurasi (%)",
                colors=["#2563eb", "#dc2626"],
                ylim=(0, 100),
            )
        )
    with col2:
        st.pyplot(
            bar_chart(
                "Perbandingan MAE",
                evaluasi["Metode"],
                evaluasi["MAE"],
                "MAE",
                colors=["#2563eb", "#dc2626"],
            )
        )

    st.subheader("Visualisasi MAE Score")
    st.caption(
        "MAE dihitung dari selisih absolut antara nilai risiko hasil model dan "
        "skor aktual kategori: RENDAH=25, SEDANG=50, TINGGI=85."
    )

    mae_per_kategori = (
        mae_scores.groupby("KATEGORI_AKTUAL")[["Mamdani", "Sugeno"]]
        .mean()
        .reindex(KATEGORI_ORDER, fill_value=0)
    )

    col_mae_a, col_mae_b = st.columns(2)
    with col_mae_a:
        st.pyplot(
            grouped_bar_chart(
                "MAE per Kategori Aktual",
                KATEGORI_ORDER,
                {
                    "Mamdani": mae_per_kategori["Mamdani"],
                    "Sugeno": mae_per_kategori["Sugeno"],
                },
                "MAE",
                colors=["#2563eb", "#dc2626"],
            )
        )
    with col_mae_b:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(
            [mae_scores["Mamdani"], mae_scores["Sugeno"]],
            labels=["Mamdani", "Sugeno"],
            patch_artist=True,
            boxprops={"facecolor": "#bfdbfe", "alpha": 0.8},
            medianprops={"color": "#111827", "linewidth": 2},
        )
        ax.set_title("Sebaran Absolute Error")
        ax.set_ylabel("Absolute Error")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)

    st.write("Rata-rata MAE per kategori aktual")
    st.dataframe(mae_per_kategori.round(2), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Confusion Matrix Mamdani")
        cm_mamdani = pd.crosstab(
            df_result["KATEGORI_AKTUAL"],
            df_result["KATEGORI_MAMDANI"],
        ).reindex(index=KATEGORI_ORDER, columns=KATEGORI_ORDER, fill_value=0)
        st.dataframe(cm_mamdani, use_container_width=True)
    with col_b:
        st.write("Confusion Matrix Sugeno")
        cm_sugeno = pd.crosstab(
            df_result["KATEGORI_AKTUAL"],
            df_result["KATEGORI_SUGENO"],
        ).reindex(index=KATEGORI_ORDER, columns=KATEGORI_ORDER, fill_value=0)
        st.dataframe(cm_sugeno, use_container_width=True)


def section_visualization(df_result):
    st.header("8. Visualisasi Perbandingan")
    warna_mamdani = "#2563eb"
    warna_sugeno = "#dc2626"

    kategori_counts = pd.DataFrame(
        {
            "Mamdani": df_result["KATEGORI_MAMDANI"]
            .value_counts()
            .reindex(KATEGORI_ORDER, fill_value=0),
            "Sugeno": df_result["KATEGORI_SUGENO"]
            .value_counts()
            .reindex(KATEGORI_ORDER, fill_value=0),
        }
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Perbandingan Kategori Risiko")
        st.bar_chart(kategori_counts)
    with col2:
        st.subheader("Distribusi Nilai Risiko")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df_result["NILAI_MAMDANI"], bins=20, alpha=0.65, label="Mamdani")
        ax.hist(df_result["NILAI_SUGENO"], bins=20, alpha=0.55, label="Sugeno")
        ax.set_xlabel("Nilai Risiko")
        ax.set_ylabel("Jumlah Data")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)

    df_jam = df_result.copy()
    df_jam["JAM_BULAT"] = df_jam["JAM"].astype(int)
    risiko_per_jam = df_jam.groupby("JAM_BULAT")[
        ["NILAI_MAMDANI", "NILAI_SUGENO"]
    ].mean()
    st.subheader("Rata-Rata Risiko Berdasarkan Jam")
    st.line_chart(risiko_per_jam)

    top_kecamatan = (
        df_result.groupby("KECAMATAN")[["NILAI_MAMDANI", "NILAI_SUGENO"]]
        .mean()
        .sort_values("NILAI_MAMDANI", ascending=False)
        .head(10)
    )
    st.subheader("10 Kecamatan dengan Rata-Rata Risiko Tertinggi")
    st.bar_chart(top_kecamatan)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Proporsi Mamdani")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            kategori_counts["Mamdani"],
            labels=KATEGORI_ORDER,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Kategori Mamdani")
        st.pyplot(fig)
    with col_b:
        st.subheader("Proporsi Sugeno")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            kategori_counts["Sugeno"],
            labels=KATEGORI_ORDER,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Kategori Sugeno")
        st.pyplot(fig)

    st.caption(f"Warna utama: Mamdani {warna_mamdani}, Sugeno {warna_sugeno}.")


def section_interpretation(df_result):
    st.header("9. Interpretasi Hasil")
    evaluasi = get_evaluation(df_result)
    mamdani_acc = evaluasi.loc[0, "Akurasi (%)"]
    sugeno_acc = evaluasi.loc[1, "Akurasi (%)"]
    mamdani_mae = evaluasi.loc[0, "MAE"]
    sugeno_mae = evaluasi.loc[1, "MAE"]

    st.markdown(
        f"""
        - Mamdani dan Sugeno sama-sama diimplementasikan **from scratch**.
        - Mamdani memakai defuzzifikasi centroid, sehingga nilai output lebih halus.
        - Sugeno memakai weighted average, sehingga perhitungannya lebih sederhana.
        - Akurasi Mamdani saat ini: **{mamdani_acc:.2f}%**.
        - Akurasi Sugeno saat ini: **{sugeno_acc:.2f}%**.
        - MAE Mamdani: **{mamdani_mae:.2f}**.
        - MAE Sugeno: **{sugeno_mae:.2f}**.

        Threshold kategori risiko memakai revisi notebook:
        `RENDAH < 40`, `SEDANG 40-75`, dan `TINGGI >= 75`.
        Revisi ini membuat prediksi tidak lagi menjadi model konstan `SEDANG`,
        sehingga evaluasi lebih jujur meskipun akurasinya turun.
        """
    )

    st.info(
        "Catatan kritis: dataset sangat dominan kategori SEDANG. Karena itu, "
        "akurasi harus dibaca bersama MAE dan confusion matrix."
    )


def main():
    intro()

    if not DATA_PATH.exists():
        st.error(f"Dataset tidak ditemukan: {DATA_PATH}")
        st.stop()

    with st.spinner("Membaca dan menyiapkan data..."):
        df_raw = load_raw_data()
        df_clean = load_clean_data()

    navigation = st.sidebar.radio(
        "Pilih proses",
        [
            "1. Dataset",
            "2. Preprocessing",
            "3. Fungsi Keanggotaan",
            "4. Fuzzifikasi & Rule Base",
            "5. Inferensi & Defuzzifikasi",
            "6. Hasil Semua Data",
            "7. Evaluasi",
            "8. Visualisasi",
            "9. Interpretasi",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.write("Dataset")
    st.sidebar.caption(str(DATA_PATH.name))
    st.sidebar.write("Input fuzzy")
    st.sidebar.caption(", ".join(INPUT_FUZZY))

    needs_result = navigation in {
        "6. Hasil Semua Data",
        "7. Evaluasi",
        "8. Visualisasi",
        "9. Interpretasi",
    }
    df_result = None
    if needs_result:
        with st.spinner("Menghitung Mamdani dan Sugeno untuk semua data..."):
            df_result = get_result_data()

    if navigation == "1. Dataset":
        section_dataset(df_raw, df_clean)
    elif navigation == "2. Preprocessing":
        section_preprocessing(df_raw, df_clean)
    elif navigation == "3. Fungsi Keanggotaan":
        section_membership(df_clean)
    elif navigation == "4. Fuzzifikasi & Rule Base":
        section_fuzzification_rules(df_clean)
    elif navigation == "5. Inferensi & Defuzzifikasi":
        section_inference(df_clean)
    elif navigation == "6. Hasil Semua Data":
        section_results(df_result)
    elif navigation == "7. Evaluasi":
        section_evaluation(df_result)
    elif navigation == "8. Visualisasi":
        section_visualization(df_result)
    else:
        section_interpretation(df_result)


if __name__ == "__main__":
    main()
