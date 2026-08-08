import glob

for fn in glob.glob('*.py'):
    try:
        with open(fn, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'Linear' in content and ('pka' in content.lower() or 'head' in content.lower()):
                print(f"=== Found in {fn} ===")
                for line in content.split('\n'):
                    if 'nn.Linear' in line or 'regression' in line.lower() or 'head' in line.lower():
                        print("  ", line.strip())
    except Exception:
        pass
