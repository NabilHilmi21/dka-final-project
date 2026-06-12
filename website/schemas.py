import re

import pandas as pd


def parse_jam(value):
    text = str(value).strip()

    try:
        if ":" in text:
            jam, menit = text.split(":", 1)
            nilai = int(jam) + int(menit) / 60
        else:
            nilai = float(text)
    except (TypeError, ValueError):
        return 0.0

    return min(max(nilai, 0.0), 23.99)


def bersihkan_teks(value, default="TIDAK DIKETAHUI"):
    text = str(value).strip().upper()
    text = re.sub(r"\s+", " ", text)

    if text in ("", "-", "NAN", "NONE"):
        return default

    return text


# definisikan skor karakteristik
def skor_karakteristik(kode):
    kode = str(kode).upper()

    if "DEPAN-DEPAN" in kode:
        return 95
    elif "BERUNTUN/GANDA" in kode:
        return 85
    elif "DEPAN-SAMPING" in kode:
        return 80
    elif "DEPAN-BELAKANG" in kode:
        return 75
    elif "BELAKANG-SAMPING" in kode:
        return 70
    elif "SAMPING-SAMPING" in kode:
        return 50
    elif "PEJALAN-KAKI/SEJENISNYA" in kode:
        return 40

    return 30


def kategori_aktual_risiko(kode_cidera):
    kode_cidera = str(kode_cidera).upper()

    if "MD" in kode_cidera or "CT" in kode_cidera:
        return "TINGGI"
    elif "LL" in kode_cidera:
        return "SEDANG"

    return "RENDAH"


def prepare_data(path="datakecminim.csv"):
    df = pd.read_csv(path)

    df = df[
        [
            "JAM",
            "BULAN",
            "KECAMATAN",
            "PROFESI",
            "KODE_CIDERA",
            "KARAKTERISTIK LAKA",
        ]
    ].copy()

    df["JAM_ASLI"] = df["JAM"]
    df["JAM"] = df["JAM"].apply(parse_jam)
    df["BULAN"] = pd.to_numeric(df["BULAN"], errors="coerce").fillna(1)
    df["BULAN"] = df["BULAN"].clip(lower=1, upper=12)
    df["KECAMATAN"] = df["KECAMATAN"].apply(bersihkan_teks)
    df["PROFESI"] = df["PROFESI"].apply(bersihkan_teks)
    df["KODE_CIDERA"] = df["KODE_CIDERA"].apply(bersihkan_teks)
    df["KARAKTERISTIK LAKA"] = df["KARAKTERISTIK LAKA"].apply(bersihkan_teks)

    # hitung frekuensi kecamatan
    df["FREQ_KECAMATAN"] = df["KECAMATAN"].map(df["KECAMATAN"].value_counts())
    df["FREQ_PROFESI"] = df["PROFESI"].map(df["PROFESI"].value_counts())

    df["SKOR_KARAKTERISTIK"] = df["KARAKTERISTIK LAKA"].apply(
        skor_karakteristik
    )
    df["KATEGORI_AKTUAL"] = df["KODE_CIDERA"].apply(kategori_aktual_risiko)

    return df
