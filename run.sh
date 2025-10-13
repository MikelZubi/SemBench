#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

# Step 1: Download the zip file
echo "Downloading EEH.sql.zip..."
curl -L -o EEH.sql.zip https://www.ehu.eus/eeh/EEH.sql.zip

# Step 2: Unzip the file
echo "Unzipping EEH.sql.zip..."
unzip -o EEH.sql.zip
# The unzip command will extract into a directory "EEH.sql/" containing "EEH-23-01-26.sql"

# Step 3: Clone mysql2sqlite and run the conversion
echo "Cloning mysql2sqlite..."
git clone https://github.com/mysql2sqlite/mysql2sqlite.git
cd mysql2sqlite

# Make sure the script is executable
chmod +x mysql2sqlite

# Run the conversion (adjust SQL filename if different)
SQL_FILE="../EEH-23-01-26.sql"
DB_FILE="../EEH-23-01-26.db"

echo "Converting MySQL dump to SQLite..."
./mysql2sqlite "$SQL_FILE" | sqlite3 "$DB_FILE"

cd ..

# Step 4: Run Python script
echo "Running eeh2sembench.py..."
python eeh2sembench.py

echo "✅ All steps completed successfully."
