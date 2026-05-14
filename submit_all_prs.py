"""
submit_all_prs.py
Run from inside FlagGems folder.
Creates branches and prepares commits for all ready operators.
"""
import subprocess
import os

READY_OPS = {
    "relu":            "optimize/relu-triton",
    "silu":            "optimize/silu-v2",
    "embedding":       "optimize/embedding-triton",
    "groupnorm":       "optimize/groupnorm-triton",
    "gather":          "optimize/gather-triton",
    "flash_attention": "optimize/flash-attention-triton",
}

OP_FILES = {
    "relu":            "src/flag_gems/ops/relu.py",
    "silu":            "src/flag_gems/ops/silu.py",
    "embedding":       "src/flag_gems/ops/embedding.py",
    "groupnorm":       "src/flag_gems/ops/groupnorm.py",
    "gather":          "src/flag_gems/ops/gather.py",
    "flash_attention": "src/flag_gems/ops/flash_attention.py",
}

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR: {r.stderr[:100]}")
        return False
    return True

print("Checking current branch...")
r = subprocess.run("git branch --show-current", shell=True, capture_output=True, text=True)
print(f"Current branch: {r.stdout.strip()}")

for op, branch in READY_OPS.items():
    src = os.path.join("..", "flaggems-optimizer", "operators", "optimized", f"{op}_optimized.py")
    dest = OP_FILES[op]
    if not os.path.exists(src):
        print(f"SKIP {op} - source file not found at {src}")
        continue
    print(f"\n--- {op} ---")
    print(f"  Source: {src}")
    print(f"  Dest:   {dest}")
    print(f"  Branch: {branch}")
    print(f"  Run these commands:")
    print(f"    git checkout master && git pull origin master")
    print(f"    git checkout -b {branch}")
    print(f"    copy \"{src.replace('/', chr(92))}\" \"{dest.replace('/', chr(92))}\"")
    print(f"    python fix_{op}.py  (if available)")
    print(f"    pre-commit run --all-files")
    print(f"    git add {dest}")
    print(f"    git commit -m \"perf({op}): fused Triton kernel\"")
    print(f"    git push origin {branch}")
