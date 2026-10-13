from pathlib import Path
import datetime
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def identity(path, force=False):
    p=Path(path).resolve(); s=p.stat()
    full=force or s.st_size<=16*1024*1024
    return {"path":str(p),"size_bytes":s.st_size,"mtime_ns":s.st_mtime_ns,
            "sha256":sha256(p) if full else None,
            "checksum_policy":"FULL_SHA256" if full else "METADATA_ONLY"}

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")

def write_jsonl(path,rows):
    with open(path,"w") as handle:
        for row in rows:
            handle.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")

def record(out,started,runtime,parameters,inputs,status="SUCCESS",error=None):
    code=Path(__file__).resolve().parents[1]
    root=code.parents[1]
    result=subprocess.run(["git","--no-optional-locks","-C",str(root),"rev-parse","HEAD"],capture_output=True,text=True)
    commit=result.stdout.strip() if result.returncode==0 else None
    dirty=None
    if commit:
        dirty=bool(subprocess.run(["git","--no-optional-locks","-C",str(root),"status","--porcelain"],capture_output=True,text=True).stdout)
    outputs=[identity(p,force=True) for p in sorted(Path(out).iterdir()) if p.is_file() and p.name!="run_record.json"]
    obj={"schema_version":"scape2-run-v1","host":socket.gethostname(),"date_utc":started,
         "completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "git_commit":commit,"git_dirty":dirty,"git_status":"RECORDED" if commit else "NOT_A_GIT_REPOSITORY",
         "environment":{"executable":sys.executable,"prefix":sys.prefix,"base_prefix":sys.base_prefix,
                        "platform":platform.platform(),"python_version":platform.python_version(),
                        "packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}},
         "input_paths":[str(Path(p).resolve()) for p in inputs],
         "inputs":[identity(p) for p in inputs],"parameters":parameters,
         "random_seed":parameters.get("seed",0),"outputs":outputs,"runtime_seconds":runtime,
         "source_sha256":{str(p.relative_to(code)):sha256(p) for p in sorted(code.rglob("*.py"))},
         "status":status,"error":error}
    write_json(Path(out)/"run_record.json",obj)
    return obj
