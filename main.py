from pathlib import Path

import matplotlib

from models import kategori_risiko, mamdani, sugeno
from schemas import prepare_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def hitung_risiko(row):
    jam = row["JAM"]
    frekuensi = row["FREQ_KECAMATAN"]
    cidera = row["SKOR_CIDERA"]
    karakteristik = row["SKOR_KARAKTERISTIK"]

    nilai_mamdani = mamdani(jam, frekuensi, cidera, karakteristik)
    nilai_sugeno = sugeno(jam, frekuensi, cidera, karakteristik)

    return {
        "NILAI_MAMDANI": round(nilai_mamdani, 2),
        "KATEGORI_MAMDANI": kategori_risiko(nilai_mamdani),
        "NILAI_SUGENO": round(nilai_sugeno, 2),
        "KATEGORI_SUGENO": kategori_risiko(nilai_sugeno),
    }


def buat_visualisasi(df):
    output_dir = Path("visualisasi")
    output_dir.mkdir(exist_ok=True)

    warna_mamdani = "#2563eb"
    warna_sugeno = "#dc2626"
    warna_kategori = {
        "RENDAH": "#16a34a",
        "SEDANG": "#f59e0b",
        "TINGGI": "#dc2626",
    }
    kategori_order = ["RENDAH", "SEDANG", "TINGGI"]

    counts = {
        "Mamdani": df["KATEGORI_MAMDANI"].value_counts().reindex(
            kategori_order, fill_value=0
        ),
        "Sugeno": df["KATEGORI_SUGENO"].value_counts().reindex(
            kategori_order, fill_value=0
        ),
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(kategori_order))
    ax.bar(
        [i - 0.18 for i in x],
        counts["Mamdani"],
        width=0.36,
        label="Mamdani",
        color=warna_mamdani,
    )
    ax.bar(
        [i + 0.18 for i in x],
        counts["Sugeno"],
        width=0.36,
        label="Sugeno",
        color=warna_sugeno,
    )
    ax.set_title("Perbandingan Kategori Risiko")
    ax.set_xlabel("Kategori Risiko")
    ax.set_ylabel("Jumlah Data")
    ax.set_xticks(list(x), kategori_order)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "kategori_risiko.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        df["NILAI_MAMDANI"],
        bins=20,
        alpha=0.65,
        label="Mamdani",
        color=warna_mamdani,
    )
    ax.hist(
        df["NILAI_SUGENO"],
        bins=20,
        alpha=0.55,
        label="Sugeno",
        color=warna_sugeno,
    )
    ax.set_title("Distribusi Nilai Risiko")
    ax.set_xlabel("Nilai Risiko")
    ax.set_ylabel("Jumlah Data")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "distribusi_nilai_risiko.png", dpi=160)
    plt.close(fig)

    df_jam = df.copy()
    df_jam["JAM_BULAT"] = df_jam["JAM"].astype(int)
    risiko_per_jam = df_jam.groupby("JAM_BULAT")[
        ["NILAI_MAMDANI", "NILAI_SUGENO"]
    ].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        risiko_per_jam.index,
        risiko_per_jam["NILAI_MAMDANI"],
        marker="o",
        label="Mamdani",
        color=warna_mamdani,
    )
    ax.plot(
        risiko_per_jam.index,
        risiko_per_jam["NILAI_SUGENO"],
        marker="o",
        label="Sugeno",
        color=warna_sugeno,
    )
    ax.set_title("Rata-Rata Risiko Berdasarkan Jam")
    ax.set_xlabel("Jam")
    ax.set_ylabel("Rata-Rata Nilai Risiko")
    ax.set_xticks(range(0, 24))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "risiko_per_jam.png", dpi=160)
    plt.close(fig)

    top_kecamatan = (
        df.groupby("KECAMATAN")[["NILAI_MAMDANI", "NILAI_SUGENO"]]
        .mean()
        .sort_values("NILAI_MAMDANI", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    top_kecamatan[["NILAI_MAMDANI", "NILAI_SUGENO"]].plot(
        kind="barh",
        ax=ax,
        color=[warna_mamdani, warna_sugeno],
    )
    ax.set_title("10 Kecamatan dengan Rata-Rata Risiko Tertinggi")
    ax.set_xlabel("Rata-Rata Nilai Risiko")
    ax.set_ylabel("Kecamatan")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "top_kecamatan_risiko.png", dpi=160)
    plt.close(fig)

    kategori_mamdani = df["KATEGORI_MAMDANI"].value_counts().reindex(
        kategori_order, fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        kategori_mamdani,
        labels=kategori_order,
        autopct="%1.1f%%",
        startangle=90,
        colors=[warna_kategori[kategori] for kategori in kategori_order],
    )
    ax.set_title("Proporsi Kategori Risiko Mamdani")
    fig.tight_layout()
    fig.savefig(output_dir / "proporsi_mamdani.png", dpi=160)
    plt.close(fig)

    return output_dir


def main():
    df = prepare_data("datakecminim.csv")

    hasil_risiko = df.apply(hitung_risiko, axis=1, result_type="expand")
    df = df.join(hasil_risiko)

    kolom_output = [
        "JAM_ASLI",
        "KECAMATAN",
        "FREQ_KECAMATAN",
        "KODE_CIDERA",
        "KARAKTERISTIK LAKA",
        "NILAI_MAMDANI",
        "KATEGORI_MAMDANI",
        "NILAI_SUGENO",
        "KATEGORI_SUGENO",
    ]

    print("\nHASIL PERHITUNGAN RISIKO KECELAKAAN")
    print("=" * 80)
    print(df[kolom_output].head(20).to_string(index=False))

    print("\nRINGKASAN KATEGORI MAMDANI")
    print(df["KATEGORI_MAMDANI"].value_counts().to_string())

    print("\nRINGKASAN KATEGORI SUGENO")
    print(df["KATEGORI_SUGENO"].value_counts().to_string())

    output_dir = buat_visualisasi(df)
    print(f"\nVisualisasi disimpan di folder: {output_dir}")


if __name__ == "__main__":
    main()
