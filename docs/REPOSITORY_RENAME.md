# Repository identity update

Current display name: **Imbalanced Learning Reproductions**.

Prepared GitHub name: `imbalanced-learning-reproductions`.

Prepared About description:

> Audited Python implementations and controlled studies of published resampling methods: UGO, BFCH-SMOTE, EC-EBO and SSHR.

## Hosting status

On 2026-10-13, ordinary Git fetch/push worked, but authenticated GitHub API
requests for repository metadata and the requested rename were blocked with
`Forbidden`. The rename command did not succeed. The actual GitHub repository
and origin therefore remain `MAYANK-RATHVA05/rb_ugo`; current clone/source links
continue using that verified identity. This is not a local-folder rename.

The README, project scope, dataset guide and contributor instructions are
prepared independently of that hosting limitation. The complete working branch
and main can display the new project description without merging implementation
changes. SSHR code remains on `sshr-implementation`.

## Finish the rename in GitHub

1. Open [repository settings](https://github.com/MAYANK-RATHVA05/rb_ugo/settings).
2. Under **General → Repository name**, enter
   `imbalanced-learning-reproductions` and click **Rename**.
3. On the repository homepage, open **About → Edit** and save the description
   above.

After GitHub confirms the rename, update an existing Windows or Linux checkout:

```bash
git remote set-url origin https://github.com/MAYANK-RATHVA05/imbalanced-learning-reproductions.git
git fetch origin
git remote -v
```

Your `C:\ugo\ugo_base` or `/workspace/rb_ugo` folder can keep its current name;
paths are relative to the repository root. Update saved cloud repository
settings to the verified renamed identity when configuring future checkouts.

The requested rename does not merge feature branches or change the default
branch. Choose `sshr-implementation` in GitHub's branch selector to inspect the
full implementation, reports and supervisor briefing. Merging its code into
main is separate work.
