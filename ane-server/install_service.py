"""Install a per-user launchd service from a validated server configuration."""
import argparse,json,os,plistlib,subprocess
from pathlib import Path
from serve import command

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--label",default="local.coreai.ane-server")
    args=parser.parse_args()
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for c in args.label):
        parser.error("label must contain only letters, digits, dots and hyphens")
    argv=command(args.config)
    config=json.loads(args.config.read_text())
    protected=[Path.home()/name for name in ("Documents","Desktop","Downloads")]
    runtime_paths=[Path(argv[0]).resolve(),Path(argv[argv.index("--model")+1]).resolve()]
    if any(path.is_relative_to(folder) for path in runtime_paths for folder in protected):
        parser.error("launchd cannot inherit terminal folder approval; keep the binary and model in Workspace")
    if config.get("api_key_env"):
        parser.error("For authenticated deployments, supply the secret through your own supervisor; this installer never persists secrets")
    domain=f"gui/{os.getuid()}"
    subprocess.run(["launchctl","print",domain],check=True,stdout=subprocess.DEVNULL)
    # Refuse replacing an existing service or plist; inspect/backup it explicitly.
    target=Path.home()/"Library/LaunchAgents"/(args.label+".plist")
    if target.exists():
        parser.error(f"Existing service configuration: {target}")
    result=Path(__file__).resolve().parent/"results"
    result.mkdir(parents=True,exist_ok=True)
    spec={"Label":args.label,"ProgramArguments":argv,
          "WorkingDirectory":str(Path(__file__).resolve().parent.parent),
          "RunAtLoad":True,"KeepAlive":True,"ThrottleInterval":30,
          "StandardOutPath":str(result/"service.stdout.log"),
          "StandardErrorPath":str(result/"service.stderr.log")}
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("xb") as stream:plistlib.dump(spec,stream)
    try:
        subprocess.run(["launchctl","bootstrap",domain,str(target)],check=True)
    except BaseException:
        target.unlink()
        raise
    print(json.dumps({"plist":str(target),"service":domain+"/"+args.label,"argv":argv}))
if __name__=="__main__":main()
