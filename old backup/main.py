from pathlib import Path

import matplotlib

from models import contoh_rules, jumlah_rules, kategori_risiko, mamdani, sugeno
from schemas import prepare_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SKOR_AKTUAL = {
    "RENDAH": 25,
    "SEDANG": 50,
    "TINGGI": 85,
}


def hitung_risiko(row):
    jam = row["JAM"]
    bulan = row["BULAN"]
    freq_kecamatan = row["FREQ_KECAMATAN"]
    freq_profesi = row["FREQ_PROFESI"]
    karakteristik = row["SKOR_KARAKTERISTIK"]

    nilai_mamdani = mamdani(
        jam,
        bulan,
        freq_kecamatan,
        freq_profesi,
        karakteristik,
    )
    nilai_sugeno = sugeno(
        jam,
        bulan,
        freq_kecamatan,
        freq_profesi,
        karakteristik,
    )

    return {
        "NILAI_MAMDANI": round(nilai_mamdani, 2),
        "KATEGORI_MAMDANI": kategori_risiko(nilai_mamdani),
        "NILAI_SUGENO": round(nilai_sugeno, 2),
        "KATEGORI_SUGENO": kategori_risiko(nilai_sugeno),
    }


def hitung_akurasi_model(df, kolom_prediksi):
    total = len(df)
    jumlah_benar = (df[kolom_prediksi] == df["KATEGORI_AKTUAL"]).sum()
    akurasi = jumlah_benar / total * 100 if total else 0

    return {
        "JUMLAH_BENAR": jumlah_benar,
        "TOTAL_DATA": total,
        "AKURASI": round(akurasi, 2),
    }


def hitung_mae_model(df, kolom_nilai):
    nilai_aktual = df["KATEGORI_AKTUAL"].map(SKOR_AKTUAL)
    error_absolut = (df[kolom_nilai] - nilai_aktual).abs()

    return round(error_absolut.mean(), 2)


def evaluasi_akurasi(df):
    return {
        "Mamdani": hitung_akurasi_model(df, "KATEGORI_MAMDANI"),
        "Sugeno": hitung_akurasi_model(df, "KATEGORI_SUGENO"),
    }


def evaluasi_mae(df):
    return {
        "Mamdani": hitung_mae_model(df, "NILAI_MAMDANI"),
        "Sugeno": hitung_mae_model(df, "NILAI_SUGENO"),
    }


def print_perbandingan_akurasi(hasil_akurasi):
    print("\nPERBANDINGAN AKURASI PREDIKSI RISIKO")
    print("=" * 80)
    print("Label aktual dihitung dari KODE_CIDERA")
    print("MD/CT = TINGGI, LL = SEDANG, selain itu = RENDAH\n")
    print("MODEL     BENAR  TOTAL  AKURASI")
    print("-" * 32)

    for model, hasil in hasil_akurasi.items():
        print(
            f"{model:<8}  "
            f"{hasil['JUMLAH_BENAR']:>5}  "
            f"{hasil['TOTAL_DATA']:>5}  "
            f"{hasil['AKURASI']:>6.2f}%"
        )

    selisih = hasil_akurasi["Sugeno"]["AKURASI"] - hasil_akurasi["Mamdani"]["AKURASI"]

    if selisih > 0:
        print(f"\nSugeno lebih akurat {selisih:.2f}% dibanding Mamdani.")
    elif selisih < 0:
        print(f"\nMamdani lebih akurat {abs(selisih):.2f}% dibanding Sugeno.")
    else:
        print("\nAkurasi Mamdani dan Sugeno sama.")


def print_perbandingan_mae(hasil_mae):
    print("\nPERBANDINGAN MAE NILAI RISIKO")
    print("=" * 80)
    print("Skor aktual: RENDAH=25, SEDANG=50, TINGGI=85")
    print("MODEL       MAE")
    print("-" * 18)

    for model, mae in hasil_mae.items():
        print(f"{model:<8}  {mae:>6.2f}")

    selisih = hasil_mae["Sugeno"] - hasil_mae["Mamdani"]

    if selisih > 0:
        print(f"\nMamdani memiliki MAE lebih kecil {selisih:.2f} poin.")
    elif selisih < 0:
        print(f"\nSugeno memiliki MAE lebih kecil {abs(selisih):.2f} poin.")
    else:
        print("\nMAE Mamdani dan Sugeno sama.")


def print_rule_base():
    print("\nRINGKASAN RULE BASE")
    print("=" * 80)
    print(f"Jumlah rule yang digunakan: {jumlah_rules()}")
    print("Contoh 15 rule pertama:")

    for index, rule in enumerate(contoh_rules(15), start=1):
        print(
            f"{index:>2}. IF jam={rule['jam']} AND bulan={rule['bulan']} "
            f"AND freq_kecamatan={rule['freq_kecamatan']} "
            f"AND freq_profesi={rule['freq_profesi']} "
            f"AND karakteristik={rule['karakteristik']} "
            f"THEN risiko={rule['output']}"
        )


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

    hasil_akurasi = evaluasi_akurasi(df)
    model_names = list(hasil_akurasi.keys())
    nilai_akurasi = [hasil_akurasi[model]["AKURASI"] for model in model_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        model_names,
        nilai_akurasi,
        color=[warna_mamdani, warna_sugeno],
        width=0.5,
    )
    ax.set_title("Perbandingan Akurasi Prediksi Risiko")
    ax.set_xlabel("Metode Fuzzy")
    ax.set_ylabel("Akurasi (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)

    for bar, nilai in zip(bars, nilai_akurasi):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{nilai:.2f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(output_dir / "akurasi_model.png", dpi=160)
    plt.close(fig)

    hasil_mae = evaluasi_mae(df)
    nilai_mae = [hasil_mae[model] for model in model_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        model_names,
        nilai_mae,
        color=[warna_mamdani, warna_sugeno],
        width=0.5,
    )
    ax.set_title("Perbandingan MAE Nilai Risiko")
    ax.set_xlabel("Metode Fuzzy")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", alpha=0.25)

    for bar, nilai in zip(bars, nilai_mae):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{nilai:.2f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(output_dir / "mae_model.png", dpi=160)
    plt.close(fig)

    return output_dir


def main():
    df = prepare_data("datakecminim.csv")

    print("SUMBER DATASET")
    print("=" * 80)
    print("Kaggle - data kecelakaan")
    print("https://www.kaggle.com/datasets/aginanjar/data-kecelakaan")
    print(f"Jumlah data: {len(df)} row")
    print("Input fuzzy: JAM, BULAN, FREQ_KECAMATAN, FREQ_PROFESI, SKOR_KARAKTERISTIK")
    print("Output/ground truth: KODE_CIDERA -> KATEGORI_AKTUAL")
    print_rule_base()

    hasil_risiko = df.apply(hitung_risiko, axis=1, result_type="expand")
    df = df.join(hasil_risiko)

    kolom_output = [
        "JAM_ASLI",
        "BULAN",
        "KECAMATAN",
        "FREQ_KECAMATAN",
        "PROFESI",
        "FREQ_PROFESI",
        "KODE_CIDERA",
        "KARAKTERISTIK LAKA",
        "KATEGORI_AKTUAL",
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

    hasil_akurasi = evaluasi_akurasi(df)
    print_perbandingan_akurasi(hasil_akurasi)
    hasil_mae = evaluasi_mae(df)
    print_perbandingan_mae(hasil_mae)

    output_dir = buat_visualisasi(df)
    print(f"\nVisualisasi disimpan di folder: {output_dir}")


if __name__ == "__main__":
    main()
