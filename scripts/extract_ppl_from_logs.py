from pathlib import Path
import re
import csv

log_files = list(Path("final_artifacts").rglob("*.out")) + list(Path("final_artifacts").rglob("*.txt")) + list(Path("final_artifacts").rglob("*.log"))

rows = []
pattern = re.compile(r"INFO\s+(wikitext2|c4)\s*:\s*([0-9.]+)", re.IGNORECASE)

for path in log_files:
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        continue

    for dataset, value in pattern.findall(text):
        rows.append({
            "log_file": str(path),
            "dataset": dataset.lower(),
            "ppl": value,
        })

Path("results").mkdir(exist_ok=True)

with open("results/extracted_ppl_from_logs.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["log_file", "dataset", "ppl"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Extracted {len(rows)} PPL records.")
for row in rows:
    print(row)
