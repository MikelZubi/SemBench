import html
import json
import re
import sqlite3
from collections import Counter


def clean_definition_text(text):
    if not text:
        return ''
    cleaned = text.split(' ik ')[0]
    cleaned = cleaned.split('_i_(ikus')[0]
    cleaned = re.sub(r'_oharra2_.+?_/oharra2_\s*$', '', cleaned)
    cleaned = cleaned.replace('_b_-_/b_', '-')
    cleaned = re.sub(r'\._([^ib/])', r'. \1', cleaned)
    cleaned = cleaned.replace('_i_(Platanus acerifolia_/i_ edo _i_hibrida)_/i_.', '(Platanus acerifolia edo hibrida).')  # ad-hoc
    cleaned = cleaned.replace('_i_(Corallimorpha_/i_ ord., _i_Madreporaria_/i_ ord. etab.). ord. etab.).', '(Corallimorpha ord., Madreporaria ord. eta abar).')  # ad-hoc
    cleaned = cleaned.replace('_i_(Cs;_/i_', '(Cs;')  # ad-hoc
    cleaned = re.sub(r'_oharra2_\(distantzietan\)_/oharra2_$', '', cleaned)  # ad-hoc
    cleaned = cleaned.replace('_i_adlag_/i_ erdizka.', '*adlag* erdizka.')  # ad-hoc
    cleaned = cleaned.replace('_b_&middot;_/b_', '')
    cleaned = re.sub('_i_irud/hed.*?_/i_', '', cleaned)
    cleaned = re.sub(r'_i_\(([^)]+)_/i_ (etab\.|eta beste)\)\.?$', r'(\1 eta abar).', cleaned)
    cleaned = re.sub(r'_i_(\([^)]+\))\.?_/i_\.?$', r'\1.', cleaned)
    cleaned = cleaned.split('_')[-1]
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def clean_example_text(text):
    if not text:
        return []
    text = html.unescape(text)
    examples = text.split('-:-')
    records = []
    for example in examples:
        example_id = None
        m = re.search(r'<span class="adKount">([^<]*)</span>', example)
        if m:
            example_id = m.group(1).strip() or None
            example = example.split('<span')[0].strip()
            example = example.replace(' i_i_k _/i__u_koordinatugabe_/u_.', '')  # ad-hoc
            example = re.split(r'\.(_i_)?\s*ik\s*_?', example)[0].strip()
            example = example.replace('_i_', '<i>').replace('_/i_', '</i>')
            example = re.sub(r'<i>([ .,:]+)', r'\1<i>', example)
            example = re.sub(r'([ .,:]+)</i>', r'</i>\1', example)
            example = example.replace('<i></i>', '')
            example = example.replace('<i>qu</i>i', '<i>qui</i>')  # ad-hoc
        example = re.sub(r'\s+', ' ', example).strip()
        records.append({'ref': example_id, 'example': example})
    return records


def extract_redirects(definition_text):
    if not definition_text:
        return None
    m = re.search(r'ik (?:_u_|_b_)+([^_]+)', definition_text)
    if not m:
        return None
    redirection = m.group(1).strip()
    # normalize: remove all internal whitespace (turn 'bai 10' -> 'bai10')
    redirection = re.sub(r'\s+', '', redirection)
    if redirection == '':
        return None
    return redirection


DB_PATH = 'EEH-23-01-26.db'
OUT_JSON = 'EEH-23-01-26.sembench.jsonl'
TABLE = 'eeh'

# --- load original table ---
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute(f'SELECT id, sarrera, sarreraH, ik, bistan, adibideak FROM {TABLE}')
rows = cur.fetchall()

# Build original full records and a lookup by sense
dictionary = []
entry_by_sense = {}
for id_, word_form, sense, redirect, bistan, examples in rows:
    if not (word_form and sense):
        continue
    record = {'id': id_, 'wf': word_form, 'sense': sense, 'ik': redirect, 'definition': bistan, 'examples': examples}
    dictionary.append(record)
    entry_by_sense[sense] = record

# --- filtering pipeline ---
filtered_dictionary = []
for record in dictionary:
    definition = record['definition']
    # 1) Keep only verbs (_i_ad_) or nouns (_i_iz_) and record pos
    pos = None
    if '_i_ad_' in definition:
        pos = 'ad'
    elif '_i_iz_' in definition:
        pos = 'iz'
    elif '_i_iz ' in definition:
        pos = 'iz'
    else:
        continue
    # 2) senses of polysemous words must have a number at the end
    if not re.search(r'\d+$', record['sense']):
        continue
    new_record = record.copy()
    new_record['pos'] = pos
    filtered_dictionary.append(new_record)

# --- resolve redirects (ik == 1) by looking up the reference in the original table ---
to_remove = []
for record in filtered_dictionary:

    if record.get('ik') == 0:
        continue

    reference = extract_redirects(record['definition'])
    if not reference:
        # no redirect possible found, exclude this entry
        to_remove.append(record['sense'])
        continue

    # First try the in-memory original lookup
    target = entry_by_sense.get(reference)
    if target:
        record['definition'] = target['definition']
        continue

    # fallback: query DB directly
    cur.execute(f'SELECT bistan FROM {TABLE} WHERE sarreraH = ? LIMIT 1', (reference,))
    row = cur.fetchone()
    if row:
        record['definition'] = row[0]
        continue

    # extra fallback: try matching by wordform (no number)
    reference = ''.join(char for char in reference if not char.isdigit())
    cur.execute('SELECT bistan, sarreraH FROM eeh WHERE sarrera = ? LIMIT 1', (reference,))
    row = cur.fetchone()
    if row:
        record['definition'] = row[0]
        continue

    # no redirect possible found, exclude this entry
    to_remove.append(record['sense'])

filtered_dictionary = [record for record in filtered_dictionary if record['sense'] not in to_remove]

# --- clean kept entries ---
for record in filtered_dictionary:
    record['definition'] = clean_definition_text(record['definition'])
    record['examples'] = clean_example_text(record['examples'])
filtered_dictionary = [record for record in filtered_dictionary if record['definition']]

# --- filter again ---
wf_counts = Counter(r['wf'] for r in filtered_dictionary)
filtered_dictionary = [record for record in filtered_dictionary if wf_counts[record['wf']] > 1]

# --- dump ---

sem_bench = {}
for record in filtered_dictionary:
    if record['wf'] not in sem_bench:
        sem_bench[record['wf']] = dict(id=record['id'], wf=record['wf'], senses=[])
    sense_record = dict(id=record['sense'], pos=record['pos'], definition=record['definition'], examples=record['examples'])
    sem_bench[record['wf']]['senses'].append(sense_record)

with open(OUT_JSON, 'w', encoding='utf-8') as wf:
    for v in sem_bench.values():
        wf.write(json.dumps(v, ensure_ascii=False) + '\n')

print(f'✅ Exported {len(sem_bench)} entries -> {OUT_JSON}')

conn.close()