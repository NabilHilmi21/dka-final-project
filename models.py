# membership function
def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0
    
    if x == b:
        return 1
    
    if x < b:
        return (x-a)/(b-a)
    
    return (c-x) / (c-b)

# membership jam (waktu)
def jam_sepi(x): return triangular(x, -1, 3, 7)
def jam_normal(x): return triangular(x, 6, 12, 17)
def jam_sibuk(x): return triangular(x, 15, 19, 24)

# membership frequensi kecamatan
def freq_rendah(x): return triangular(x, 0, 50, 150)
def freq_sedang(x): return triangular(x, 100, 300, 500)
def freq_tinggi(x): return triangular(x, 400, 700, 1000)

# membership cidera
def cidera_ringan(x): return triangular(x, 0, 30, 50)
def cidera_sedang(x): return triangular(x, 40, 60, 80)
def cidera_berat(x): return triangular(x, 70, 100, 120)

# membership karakteristik
def karakteristik_rendah(x): return triangular(x, 0, 40, 70)
def karakteristik_sedang(x): return triangular(x, 50, 70, 90)
def karakteristik_tinggi(x): return triangular(x, 75, 95, 100)

# fuzzification
def fuzzification(jam, freq, cidera, karakteristik):
    return {
        "jam_sepi": jam_sepi(jam),
        "jam_normal": jam_normal(jam),
        "jam_sibuk": jam_sibuk(jam),

        "freq_rendah": freq_rendah(freq),
        "freq_sedang": freq_sedang(freq),
        "freq_tinggi": freq_tinggi(freq),

        "cidera_ringan": cidera_ringan(cidera),
        "cidera_sedang": cidera_sedang(cidera),
        "cidera_berat": cidera_berat(cidera),

        "karakteristik_rendah": karakteristik_rendah(karakteristik),
        "karakteristik_sedang": karakteristik_sedang(karakteristik),
        "karakteristik_tinggi": karakteristik_tinggi(karakteristik)
    }
    
# definisikan rulesnya
def rules(f):
    jam_set = [
        ("sepi", f["jam_sepi"]),
        ("normal", f["jam_normal"]),
        ("sibuk", f["jam_sibuk"]),
    ]
    freq_set = [
        ("rendah", f["freq_rendah"]),
        ("sedang", f["freq_sedang"]),
        ("tinggi", f["freq_tinggi"]),
    ]
    cidera_set = [
        ("ringan", f["cidera_ringan"]),
        ("sedang", f["cidera_sedang"]),
        ("berat", f["cidera_berat"]),
    ]
    karakteristik_set = [
        ("rendah", f["karakteristik_rendah"]),
        ("sedang", f["karakteristik_sedang"]),
        ("tinggi", f["karakteristik_tinggi"]),
    ]

    nilai_level = {
        "sepi": 0,
        "normal": 1,
        "sibuk": 2,
        "rendah": 0,
        "sedang": 1,
        "tinggi": 2,
        "ringan": 0,
        "berat": 2,
    }

    rendah = 0
    sedang = 0
    tinggi = 0

    for jam, mu_jam in jam_set:
        for freq, mu_freq in freq_set:
            for cidera, mu_cidera in cidera_set:
                for karakteristik, mu_karakteristik in karakteristik_set:
                    strength = min(
                        mu_jam,
                        mu_freq,
                        mu_cidera,
                        mu_karakteristik,
                    )

                    if strength == 0:
                        continue

                    skor = (
                        nilai_level[jam]
                        + nilai_level[freq]
                        + nilai_level[cidera] * 2
                        + nilai_level[karakteristik] * 2
                    )

                    if skor <= 3:
                        rendah = max(rendah, strength)
                    elif skor <= 6:
                        sedang = max(sedang, strength)
                    else:
                        tinggi = max(tinggi, strength)

    return rendah, sedang, tinggi

# fuzzy mamdani
def risiko_rendah(z): return triangular(z, 0, 25, 50)
def risiko_sedang(z): return triangular(z, 30, 50, 70)
def risiko_tinggi(z): return triangular(z, 60, 85, 100)

# fuzzy mamdani
def mamdani(jam, freq, cidera, karakteristik):
    f = fuzzification(jam, freq, cidera, karakteristik)

    rendah, sedang, tinggi = rules(f)

    numerator = 0
    denominator = 0

    for z in range(0, 101):
        mu = max(
            min(rendah, risiko_rendah(z)),
            min(sedang, risiko_sedang(z)),
            min(tinggi, risiko_tinggi(z))
        )

        numerator += z * mu
        denominator += mu

    if denominator == 0:
        return 0

    return numerator / denominator

# fuzzy sugeno
def sugeno(jam, freq, cidera, karakteristik):
    f = fuzzification(jam, freq, cidera, karakteristik)

    rendah, sedang, tinggi = rules(f)

    z_rendah = 25
    z_sedang = 50
    z_tinggi = 85

    numerator = (
        rendah * z_rendah +
        sedang * z_sedang +
        tinggi * z_tinggi
    )

    denominator = rendah + sedang + tinggi

    if denominator == 0:
        return 0

    return numerator / denominator

def kategori_risiko(nilai):
    if nilai < 40:
        return "RENDAH"
    elif nilai < 70:
        return "SEDANG"
    else:
        return "TINGGI"
