<#
Interactive utility to find and optionally remove empty directories in the repo.

Usage:
  # Dry-run (default) - just list candidates
  .\dev\clean-empty-dirs.ps1

  # Prompt to delete each candidate
  .\dev\clean-empty-dirs.ps1

  # Delete all candidate directories without prompting
  .\dev\clean-empty-dirs.ps1 -DeleteAll

  # Specify a different root directory
  .\dev\clean-empty-dirs.ps1 -RootPath 'C:\path\to\repo' -DryRun

Safety notes:
- By default this script will NOT touch `.venv`, `models`, or `results` directories.
- The default mode is dry-run; pass `-DeleteAll` to remove automatically or respond `Y` when prompted.
- Always review the list before confirming deletion.
#>

param(
    [string]$RootPath = (Get-Location).Path,
    [switch]$DeleteAll,
    [switch]$DryRun
)

function Is-EmptyDirectory($dir) {
    # Consider a directory empty if it contains no files and no non-empty subdirectories
    try {
        $children = Get-ChildItem -LiteralPath $dir -Force -ErrorAction SilentlyContinue
        if (-not $children) { return $true }
        # If there are items, ensure every child is a directory and recursively empty
        foreach ($c in $children) {
            if ($c.PSIsContainer) {
                if (-not (Is-EmptyDirectory $c.FullName)) { return $false }
            } else {
                return $false
            }
        }
        return $true
    } catch {
        return $false
    }
}

Write-Host "Scanning for empty directories under: $RootPath" -ForegroundColor Cyan

# Exclude common directories we don't want to delete
$excludeNames = @('.venv','venv','models','results','.git')

$allDirs = Get-ChildItem -Path $RootPath -Directory -Recurse -Force -ErrorAction SilentlyContinue
$candidates = @()

foreach ($d in $allDirs) {
    # Skip excluded directory names anywhere in path
    $skip = $false
    foreach ($ex in $excludeNames) {
        if ($d.FullName -match [regex]::Escape("\\$ex\\")) { $skip = $true; break }
    }
    if ($skip) { continue }

    if (Is-EmptyDirectory $d.FullName) {
        $candidates += $d.FullName
    }
}

if (-not $candidates) {
    Write-Host "No empty directories found (after exclusions)." -ForegroundColor Green
    exit 0
}

Write-Host "Found $($candidates.Count) empty directories:" -ForegroundColor Yellow
$candidates | ForEach-Object { Write-Host " - $_" }

if ($DryRun) {
    Write-Host "Dry run mode; no deletions performed." -ForegroundColor Cyan
    exit 0
}

if ($DeleteAll) {
    Write-Host "Deleting all candidate directories (no prompts)." -ForegroundColor Magenta
    foreach ($dir in $candidates) {
        try {
            Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction Stop
            Write-Host "Removed: $dir" -ForegroundColor Green
        } catch {
            Write-Host "Failed to remove: $dir -- $_" -ForegroundColor Red
        }
    }
    exit 0
}

# Interactive prompt per directory
foreach ($dir in $candidates) {
    $answer = Read-Host "Delete directory? [y/N] $dir"
    if ($answer -match '^[Yy]') {
        try {
            Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction Stop
            Write-Host "Removed: $dir" -ForegroundColor Green
        } catch {
            Write-Host "Failed to remove: $dir -- $_" -ForegroundColor Red
        }
    } else {
        Write-Host "Skipped: $dir" -ForegroundColor Gray
    }
}

Write-Host "Done." -ForegroundColor Cyan
