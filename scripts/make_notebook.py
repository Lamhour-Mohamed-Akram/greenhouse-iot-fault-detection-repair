"""Convert a `# %%`-annotated script into a Jupyter notebook.

Usage: python make_notebook.py scripts/02_fault_injection.py notebooks/02_fault_injection.ipynb
"""
import json
import re
import sys


def convert(src_path, out_path):
    src = open(src_path).read()
    cells = []
    for ch in re.split(r"^# %%", src, flags=re.M):
        ch = ch.strip("\n")
        if not ch.strip():
            continue
        if ch.startswith(" [markdown]"):
            body = ch.split("\n", 1)[1] if "\n" in ch else ""
            lines = [re.sub(r"^# ?", "", l) for l in body.split("\n")]
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": [l + "\n" for l in lines]})
        else:
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                          "outputs": [], "source": [l + "\n" for l in ch.split("\n")]})
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12"}},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(out_path, "w"), indent=1)
    print(f"{out_path}: {len(cells)} cells")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
