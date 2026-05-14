"""
Targeted fix for master's softmax.py.
Only adds empty tensor guard - does NOT replace the file.
"""
import os
import re

path = os.path.join("src", "flag_gems", "ops", "softmax.py")
code = open(path, encoding="utf-8").read()

# Check if already fixed
if "numel() == 0" in code:
    print("Already has empty tensor fix - no changes needed")
else:
    # Find the softmax function and add empty tensor guard
    old = '''def softmax(
    self,
    dim=None,
    half_to_float=False,
):
    logger.debug("GEMS SOFTMAX")'''
    
    new = '''def softmax(
    self,
    dim=None,
    half_to_float=False,
):
    logger.debug("GEMS SOFTMAX")
    if self.numel() == 0:
        return self.clone()'''
    
    if old in code:
        code = code.replace(old, new)
        print("Added empty tensor fix to softmax()")
    else:
        # Try alternate signature
        old2 = 'def softmax(self, dim=None, half_to_float=False):\n    logger.debug("GEMS SOFTMAX")'
        new2 = 'def softmax(self, dim=None, half_to_float=False):\n    logger.debug("GEMS SOFTMAX")\n    if self.numel() == 0:\n        return self.clone()'
        if old2 in code:
            code = code.replace(old2, new2)
            print("Added empty tensor fix to softmax() (alt)")
        else:
            print("Could not find softmax() signature - printing first 50 lines for inspection")
            for i, line in enumerate(code.split('\n')[200:260], 200):
                print(f"{i}: {line}")

with open(path, "w", encoding="utf-8") as f:
    f.write(code)

print("\nCurrent softmax signatures found:")
for i, line in enumerate(code.split('\n')):
    if line.strip().startswith('def ') and 'softmax' in line.lower():
        print(f"  line {i}: {line.strip()}")

print("\nEmpty tensor fix present:", "numel() == 0" in code)
