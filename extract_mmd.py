"""Extract ```mermaid blocks from full_paper.md into paper/figures/figNN.mmd."""
import re, os
os.makedirs("paper/figures", exist_ok=True)
src = open("full_paper.md").read()
blocks = re.findall(r"```mermaid\n(.*?)```", src, re.DOTALL)
for i, b in enumerate(blocks, 1):
    with open(f"paper/figures/fig{i:02d}.mmd", "w") as f:
        f.write(b)
print(f"wrote {len(blocks)} .mmd files")
