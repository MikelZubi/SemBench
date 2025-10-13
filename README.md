# EEH → SQLite → SemBench Pipeline

This repository provides a simple, reproducible pipeline to:

1. Download the **Egungo Euskararen Hiztegia (EEH) SQL dataset** from the University of the Basque Country.  
2. Convert it from a MySQL dump into an **SQLite database** using the `mysql2sqlite` tool.  
3. Run a Python script to create a SemBench dictionary from EEH.

---

## 📦 Contents

- `run.sh` — Main Bash script that automates the entire workflow.
- `eeh2sembench.py` — Python script invoked at the final stage.
- Automatically created directories and files:
  - `EEH.sql.zip` — Downloaded SQL dataset (compressed).
  - `EEH-23-01-26.sql` — Extracted SQL file.
  - `EEH-23-01-26.db` — Generated SQLite database.
  - `EEH-23-01-26.sembench.jsonl` — Generated SemBench dictionary.
  - `mysql2sqlite/` — Cloned converter repository.

---

## 🧩 Requirements

Make sure you have the following tools installed:

- `curl`
- `unzip`
- `git`
- `sqlite3`
- `python3`

## 🚀 Usage

To run the full pipeline, do:

```
./run.sh
```

This will:

1. Download the EEH SQL dump from https://www.ehu.eus/eeh/EEH.sql.zip and extract it.
2. Clone the [`mysql2sqlite`](https://github.com/mysql2sqlite) repository to convert the SQL dump to SQLite format.
3. Execute the Python script to create the SemBench resource.

## ⚖️ License Information

### EEH Dataset

The EEH SQL dump and its contents are distributed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
See the full license text here: https://creativecommons.org/licenses/by/4.0/deed.eu

### Repository Code

All original code in this repository is provided under the MIT License, unless otherwise stated.