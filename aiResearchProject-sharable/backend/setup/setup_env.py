from pathlib import Path


def setup_env():
    script_path = Path(__file__)


    project_root_dir = script_path.parent.parent.parent


    env_path = project_root_dir / ".env"

    if not env_path.exists():
        env_path.parent.mkdir(parents=True, exist_ok=True)
        with open(env_path, "w") as f:
            f.write("OPENAI_API_KEY=\n")
            f.write("GOOGLE_API_KEY=\n")
    else:
        print("env file exists")
