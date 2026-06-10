from itertools import product


RISK_OUTPUT = {
    "rendah": 25,
    "sedang": 50,
    "tinggi": 85,
}


def triangular(x, a, b, c):
    x = float(x)

    if x <= a or x >= c:
        return 0.0

    if x == b:
        return 1.0

    if x < b:
        return (x - a) / (b - a)

    return (c - x) / (c - b)


def trapezoidal(x, a, b, c, d):
    x = float(x)

    if a == b and x <= b:
        return 1.0

    if c == d and x >= c:
        return 1.0

    if x <= a or x >= d:
        return 0.0

    if b <= x <= c:
        return 1.0

    if a < x < b:
        return (x - a) / (b - a)

    return (d - x) / (d - c)


# Variable linguistic input: waktu kecelakaan.
def jam_sepi(x):
    return trapezoidal(x, 0, 0, 5, 8)


def jam_normal(x):
    return trapezoidal(x, 6, 9, 15, 18)


def jam_sibuk(x):
    return trapezoidal(x, 16, 18, 23, 23)


# Variable linguistic input: periode bulan kejadian.
def bulan_awal(x):
    return trapezoidal(x, 1, 1, 3, 5)


def bulan_tengah(x):
    return trapezoidal(x, 4, 6, 8, 10)


def bulan_akhir(x):
    return trapezoidal(x, 9, 11, 12, 12)


# Variable linguistic input: frekuensi kecamatan pada dataset.
def freq_kecamatan_rendah(x):
    return trapezoidal(x, 0, 0, 50, 200)


def freq_kecamatan_sedang(x):
    return trapezoidal(x, 100, 250, 600, 1000)


def freq_kecamatan_tinggi(x):
    return trapezoidal(x, 700, 1500, 5065, 5065)


# Variable linguistic input: frekuensi profesi pada dataset.
def freq_profesi_rendah(x):
    return trapezoidal(x, 0, 0, 1000, 3500)


def freq_profesi_sedang(x):
    return trapezoidal(x, 2000, 4000, 8000, 10000)


def freq_profesi_tinggi(x):
    return trapezoidal(x, 7000, 10000, 12000, 12000)


# Variable linguistic input: karakteristik tabrakan.
def karakteristik_rendah(x):
    return trapezoidal(x, 0, 0, 40, 65)


def karakteristik_sedang(x):
    return triangular(x, 50, 70, 90)


def karakteristik_tinggi(x):
    return trapezoidal(x, 75, 90, 100, 100)


# Variable linguistic output Mamdani: risiko cidera.
def risiko_rendah(z):
    return triangular(z, 0, 25, 50)


def risiko_sedang(z):
    return triangular(z, 30, 50, 70)


def risiko_tinggi(z):
    return triangular(z, 60, 85, 100)


def fuzzification(jam, bulan, freq_kecamatan, freq_profesi, karakteristik):
    return {
        "jam": {
            "sepi": jam_sepi(jam),
            "normal": jam_normal(jam),
            "sibuk": jam_sibuk(jam),
        },
        "bulan": {
            "awal": bulan_awal(bulan),
            "tengah": bulan_tengah(bulan),
            "akhir": bulan_akhir(bulan),
        },
        "freq_kecamatan": {
            "rendah": freq_kecamatan_rendah(freq_kecamatan),
            "sedang": freq_kecamatan_sedang(freq_kecamatan),
            "tinggi": freq_kecamatan_tinggi(freq_kecamatan),
        },
        "freq_profesi": {
            "rendah": freq_profesi_rendah(freq_profesi),
            "sedang": freq_profesi_sedang(freq_profesi),
            "tinggi": freq_profesi_tinggi(freq_profesi),
        },
        "karakteristik": {
            "rendah": karakteristik_rendah(karakteristik),
            "sedang": karakteristik_sedang(karakteristik),
            "tinggi": karakteristik_tinggi(karakteristik),
        },
    }


def level_nilai():
    return {
        "sepi": 0,
        "normal": 1,
        "sibuk": 2,
        "awal": 0,
        "tengah": 1,
        "akhir": 2,
        "rendah": 0,
        "sedang": 1,
        "tinggi": 2,
    }


def skor_rule(rule, level):
    return (
        level[rule["jam"]] * 1.0
        + level[rule["bulan"]] * 0.5
        + level[rule["freq_kecamatan"]] * 1.5
        + level[rule["freq_profesi"]] * 1.0
        + level[rule["karakteristik"]] * 2.5
    )


def kategori_rule(skor):
    if skor <= 3:
        return "rendah"

    if skor <= 6:
        return "sedang"

    return "tinggi"


def generate_rule_base():
    level = level_nilai()
    variables = {
        "jam": ["sepi", "normal", "sibuk"],
        "bulan": ["awal", "tengah", "akhir"],
        "freq_kecamatan": ["rendah", "sedang", "tinggi"],
        "freq_profesi": ["rendah", "sedang", "tinggi"],
        "karakteristik": ["rendah", "sedang", "tinggi"],
    }

    rule_base = []
    for values in product(*variables.values()):
        rule = dict(zip(variables.keys(), values))
        skor = skor_rule(rule, level)
        rule["output"] = kategori_rule(skor)
        rule_base.append(rule)

    return rule_base


RULE_BASE = generate_rule_base()


def jumlah_rules():
    return len(RULE_BASE)


def contoh_rules(limit=15):
    return RULE_BASE[:limit]


def firing_strength(fuzzy_input, rule):
    return min(
        fuzzy_input["jam"][rule["jam"]],
        fuzzy_input["bulan"][rule["bulan"]],
        fuzzy_input["freq_kecamatan"][rule["freq_kecamatan"]],
        fuzzy_input["freq_profesi"][rule["freq_profesi"]],
        fuzzy_input["karakteristik"][rule["karakteristik"]],
    )


def inferensi(fuzzy_input):
    hasil = {"rendah": 0.0, "sedang": 0.0, "tinggi": 0.0}
    aktif = []

    for index, rule in enumerate(RULE_BASE, start=1):
        strength = firing_strength(fuzzy_input, rule)

        if strength == 0:
            continue

        output = rule["output"]
        hasil[output] = max(hasil[output], strength)
        aktif.append((index, rule, strength))

    return hasil, aktif


def mamdani(jam, bulan, freq_kecamatan, freq_profesi, karakteristik):
    fuzzy_input = fuzzification(
        jam,
        bulan,
        freq_kecamatan,
        freq_profesi,
        karakteristik,
    )
    hasil_inferensi, _ = inferensi(fuzzy_input)

    numerator = 0.0
    denominator = 0.0

    for z in range(0, 101):
        mu = max(
            min(hasil_inferensi["rendah"], risiko_rendah(z)),
            min(hasil_inferensi["sedang"], risiko_sedang(z)),
            min(hasil_inferensi["tinggi"], risiko_tinggi(z)),
        )
        numerator += z * mu
        denominator += mu

    if denominator == 0:
        return 0.0

    return numerator / denominator


def sugeno(jam, bulan, freq_kecamatan, freq_profesi, karakteristik):
    fuzzy_input = fuzzification(
        jam,
        bulan,
        freq_kecamatan,
        freq_profesi,
        karakteristik,
    )
    hasil_inferensi, _ = inferensi(fuzzy_input)

    numerator = sum(
        hasil_inferensi[kategori] * nilai
        for kategori, nilai in RISK_OUTPUT.items()
    )
    denominator = sum(hasil_inferensi.values())

    if denominator == 0:
        return 0.0

    return numerator / denominator


def kategori_risiko(nilai):
    if nilai < 20:
        return "RENDAH"

    if nilai < 86:
        return "SEDANG"

    return "TINGGI"
