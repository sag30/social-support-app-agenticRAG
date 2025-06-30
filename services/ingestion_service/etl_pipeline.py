import os
import json
import pandas as pd

RAW_DIR = 'data/raw'
PROC_DIR = 'data/processed'
os.makedirs(PROC_DIR, exist_ok=True)

manifest = []

for fname in os.listdir(RAW_DIR):
    path = os.path.join(RAW_DIR, fname)
    try:
        out = ingest_file(path)
        base, _ = os.path.splitext(fname)

        # 1) Multi-sheet Excel → one CSV per sheet
        if isinstance(out, dict):
            for sheet_name, df in out.items():
                out_path = os.path.join(PROC_DIR, f'{base}_{sheet_name}.csv')
                df.to_csv(out_path, index=False)
                manifest.append({
                    'source': fname,
                    'type': 'table',
                    'sheet': sheet_name,
                    'output': out_path
                })
                print(f'✔ Processed {fname} [{sheet_name}] → CSV')
        # 2) Single DataFrame (PDF table) → CSV
        elif isinstance(out, pd.DataFrame):
            out_path = os.path.join(PROC_DIR, f'{base}.csv')
            out.to_csv(out_path, index=False)
            manifest.append({
                'source': fname,
                'type': 'table',
                'output': out_path
            })
            print(f'✔ Processed {fname} → CSV')
        # 3) Text → TXT
        else:
            out_path = os.path.join(PROC_DIR, f'{base}.txt')
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(out)
            manifest.append({
                'source': fname,
                'type': 'text',
                'output': out_path
            })
            print(f'✔ Processed {fname} → TXT')

    except Exception as e:
        print(f'✘ Error on {fname}: {e}')

# Write manifest
with open(os.path.join(PROC_DIR, 'manifest.json'), 'w') as mf:
    json.dump(manifest, mf, indent=2)

print('✅ ETL pipeline complete. See data/processed/manifest.json')
