from pathlib import Path
from textwrap import wrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


SOURCE = Path("laporan_fuzzy_kecelakaan.md")
OUTPUT = Path("laporan_fuzzy_kecelakaan.pdf")


def clean_markdown(line):
    line = line.rstrip()

    if line.startswith("# "):
        return line[2:].upper(), 18, "bold"

    if line.startswith("## "):
        return line[3:], 14, "bold"

    if line.startswith("### "):
        return line[4:], 12, "bold"

    if line.startswith("- "):
        return "  - " + line[2:], 10, "normal"

    return line, 10, "normal"


def write_page(pdf, page_lines):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    y = 0.95

    for text, size, weight in page_lines:
        fig.text(
            0.08,
            y,
            text,
            fontsize=size,
            fontweight=weight,
            family="monospace" if text.startswith("|") else "sans-serif",
            va="top",
        )
        y -= 0.026 if size <= 10 else 0.038

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pages(lines):
    pages = []
    current = []
    y_units = 0

    for raw_line in lines:
        text, size, weight = clean_markdown(raw_line)
        width = 88 if size <= 10 else 72
        wrapped = wrap(text, width=width) if text else [""]

        for item in wrapped:
            cost = 2 if size > 10 else 1
            if y_units + cost > 38:
                pages.append(current)
                current = []
                y_units = 0

            current.append((item, size, weight))
            y_units += cost

    if current:
        pages.append(current)

    return pages


def main():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    pages = build_pages(lines)

    with PdfPages(OUTPUT) as pdf:
        for page in pages:
            write_page(pdf, page)

    print(f"Laporan PDF dibuat: {OUTPUT}")


if __name__ == "__main__":
    main()
