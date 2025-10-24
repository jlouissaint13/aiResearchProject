import os
def setup_env():

    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write("OPENAI_API_KEY=\n")
            f.write("GOOGLE_API_KEY=\n")
            print("env created")
    else:
            print(".env file already exists.")