#!/usr/bin/env python3
"""Check ISA target gates independently; this does not run hardware."""
import json
import os
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent
compiler=Path(os.environ['HEXAGON_TOOLS_ROOT'])/'Tools/bin/hexagon-clang'
logs=root/'logs/compile';logs.mkdir(parents=True,exist_ok=True)
summary=[]
groups=[('matrix','matrix/compile_probe.c','PROBE_KIND',6,[68]),
        ('legacy','epilogue/compile_legacy.c','PROBE_CASE',14,[68]),
        ('modern','epilogue/compile_modern.c','PROBE_CASE',2,[68,73,81])]
for family,source,macro,count,architectures in groups:
    for architecture in architectures:
        for case in range(count):
            name=f'{family}-v{architecture}-{case}'
            command=[str(compiler),f'-mv{architecture}','-mhmx','-O2','-c',f'-D{macro}={case}',
                     str(root/'compile-probes'/source),'-o',str(logs/(name+'.o'))]
            run=subprocess.run(command,capture_output=True,text=True,timeout=60)
            diagnostic=run.stdout+run.stderr
            (logs/(name+'.log')).write_text(diagnostic)
            summary.append(dict(name=name,command=command,exit=run.returncode,diagnostic=diagnostic))
            print(name,run.returncode,flush=True)
(logs/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
# The v68 rejections are expected research outcomes; they are not a build error.
