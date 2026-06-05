import pandas as pd


def parse_jam(value):
    text = str(value).strip()

    if ":" in text:
        jam, menit = text.split(":", 1)
        return int(jam) + int(menit) / 60

    return float(text)


# definikan skor cidera
def skor_cidera(kode):
    kode = str(kode).upper()

    if "MD" in kode:  # meninggal dunia
        return 100
    elif "CT" in kode:  # cidera tubuh
        return 80
    elif "LL" in kode:  # luka-luka / luka ringan
        return 70
    else:
        return 30  # property damage, dll


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


def prepare_data(path="datakecminim.csv"):
    df = pd.read_csv(path)

    df = df[["JAM", "KECAMATAN", "KODE_CIDERA", "KARAKTERISTIK LAKA"]].copy()

    df["JAM_ASLI"] = df["JAM"]
    df["JAM"] = df["JAM"].apply(parse_jam)
    df["KECAMATAN"] = df["KECAMATAN"].astype(str)
    df["KODE_CIDERA"] = df["KODE_CIDERA"].astype(str)
    df["KARAKTERISTIK LAKA"] = df["KARAKTERISTIK LAKA"].astype(str)

    # hitung frekuensi kecamatan
    df["FREQ_KECAMATAN"] = df["KECAMATAN"].map(df["KECAMATAN"].value_counts())

    df["SKOR_CIDERA"] = df["KODE_CIDERA"].apply(skor_cidera)
    df["SKOR_KARAKTERISTIK"] = df["KARAKTERISTIK LAKA"].apply(
        skor_karakteristik
    )

    return df
