# Credential Rotation & Repository Sanitization

This project previously stored real Alpaca and QuantConnect credentials in the working tree. Those values have been scrubbed in the current branch, but they still exist in older commits. Follow the steps below immediately to rotate the keys and purge the sensitive data from the Git history.

## 1. Rotate all exposed keys

1. **Alpaca (Paper or Live) Account**
   - Log in to [https://app.alpaca.markets](https://app.alpaca.markets) ? *API Keys*.
   - Revoke the existing key pair and generate a new one.
   - Store the new `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` in a secrets manager (not in Git).
2. **QuantConnect API Token**
   - Visit [https://www.quantconnect.com/secure#api](https://www.quantconnect.com/secure#api).
   - Revoke the current API token and generate a new token.
3. Update your local environment variables or `.env` file **without committing it**.

## 2. Purge secrets from Git history

> **Prerequisite:** Install `git-filter-repo` (recommended) or prepare to use `git filter-branch`. The commands below assume `git-filter-repo` is available.

1. Create a backup in case anything goes wrong:

   ```bash
   git clone --mirror <current-repo> ../algotrendy-backup
   ```

2. Remove all historical versions of `.env` and any files known to contain the keys:

   ```bash
   git filter-repo --path .env --invert-paths
   git filter-repo --path .local/state/replit/agent/filesystem/filesystem_state.json --invert-paths
   ```

   Add additional `--path` entries for any other sensitive files discovered during audit.

3. Force-push the rewritten history to the remote:

   ```bash
   git push --force --tags origin main
   ```

4. Invalidate all existing clones by notifying collaborators. They must re-clone or run the same `git filter-repo` commands locally.

## 3. Verify sanitization

1. Run a secrets scan to confirm no credentials remain:

   ```bash
   git rev-list --all | xargs git grep -n "ALPACA_SECRET_KEY"
   ```

   The command should return no matches after the rewrite.

2. Double-check that `.env`, `.local/`, and other secret-bearing files are ignored by Git.

## 4. Record the remediation

- Update `development/rd/tasks.md` to mark the credential rotation task complete.
- Note the completion date, rotated key identifiers (if needed), and the commands executed in your internal runbook/audit log.

Following these steps removes currently exposed values and prevents them from reappearing in future commits.

## Completion Log
- Completed: 2025-09-27. Credential rotation and history purge performed. New keys active.

