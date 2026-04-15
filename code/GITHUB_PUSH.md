# Push this folder to GitHub (new code repository)

1. Create an empty public repository on GitHub (e.g. `Adaptive-Suppression-of-Trigger-Word-Persistence-in-Streaming-LLM-Dialogue_Code`).
2. In PowerShell:

```text
cd "C:\Users\13511\桌面\Project RSE\eswa_code_release"
git init
git add .
git commit -m "ESWA reproducibility: experiments, prompts, src, configs, tests"
git branch -M main
git remote add origin git@github.com:YOUR_USER/YOUR_REPO_NAME.git
git push -u origin main
```

Replace `YOUR_USER/YOUR_REPO_NAME` with your GitHub username and repository name. Use SSH or HTTPS per your setup.
