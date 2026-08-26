# GitHub Upload Guide

The package is already arranged as the repository root. Do not upload the outer
ZIP as a single file; extract it first so `README.md` appears on the GitHub home
page.

## Browser method

1. Sign in to the `Mind-Twister-Wizard` GitHub account.
2. Create a **public** repository named exactly `HumanGate-Q`.
3. Do not initialize it with another README, `.gitignore`, or license.
4. Extract the delivered ZIP and open its inner `HumanGate-Q` folder.
5. On the empty repository page, choose **uploading an existing file**, drag in
   the folder contents, and commit them.
6. Open the public repository in an incognito window and check the README,
   architecture image, CSV links, and citation panel.

GitHub's browser uploader may be inconvenient for a nested project. The command
line is more reliable.

## Command-line method

From inside the extracted `HumanGate-Q` folder:

```bash
git init
git add .
git commit -m "Release HumanGate-Q reproducibility package v2.1.0"
git branch -M main
git remote add origin https://github.com/Mind-Twister-Wizard/HumanGate-Q.git
git push -u origin main
```

Then create a release from GitHub's **Releases** page:

- tag: `v2.1.0`;
- title: `HumanGate-Q v2.1.0 — chapter reproducibility release`;
- description: state that the release contains the implementation and audited
  results supporting the submitted chapter.

## Final checks

- `data/raw/` contains only its README in the public repository.
- No `kaggle.json`, password, API token, private review, or unpublished DOCX is
  present.
- `results/paper_run/` is visible and contains the archived evidence.
- GitHub Actions completes successfully.
- The chapter uses the exact public URL in
  `CHAPTER_REPOSITORY_STATEMENT.md`.
