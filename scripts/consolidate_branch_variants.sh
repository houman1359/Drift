#!/usr/bin/env bash
set -euo pipefail

# Consolidate unique/different .py files from all branches into
# src/python/branch_variants/<branch>/ preserving paths.

cd "$(git rev-parse --show-toplevel)"

current_branch=$(git rev-parse --abbrev-ref HEAD)
root_dir="src/python/branch_variants"
report="$root_dir/REPORT.md"
mkdir -p "$root_dir"
: > "$report"

echo "Fetching branches..."
git fetch --all --prune --quiet || true

git for-each-ref --format="%(refname:short)" refs/heads refs/remotes \
  | sed 's#^origin/##' \
  | sort -u > /tmp/all_branches.txt

added_files=0
while IFS= read -r br; do
  [ -z "$br" ] && continue
  [ "$br" = "HEAD" ] && continue

  if git show-ref --verify --quiet "refs/heads/$br"; then
    ref="$br"
  elif git show-ref --verify --quiet "refs/remotes/origin/$br"; then
    ref="origin/$br"
  else
    continue
  fi

  echo "" | tee -a "$report"
  echo "=== $br ===" | tee -a "$report"

  pylist=$(git ls-tree -r --name-only "$ref" -- '*.py' || true)
  if [ -z "$pylist" ]; then
    echo "(no py files)" | tee -a "$report"
    continue
  fi

  safe_branch=${br//\//__}
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    if git show "$current_branch:$f" >/dev/null 2>&1; then
      if git diff --quiet "$current_branch:$f" "$ref:$f"; then
        echo "SAME    $f" | tee -a "$report"
      else
        echo "DIFF    $f" | tee -a "$report"
        mkdir -p "$root_dir/$safe_branch/$(dirname "$f")"
        git show "$ref:$f" > "$root_dir/$safe_branch/$f"
        added_files=$((added_files+1))
      fi
    else
      echo "NEW     $f" | tee -a "$report"
      mkdir -p "$root_dir/$safe_branch/$(dirname "$f")"
      git show "$ref:$f" > "$root_dir/$safe_branch/$f"
      added_files=$((added_files+1))
    fi
  done <<< "$pylist"
done < /tmp/all_branches.txt

echo "" | tee -a "$report"
echo "Total new/variant files added: $added_files" | tee -a "$report"

echo "Current branch: $current_branch"
echo "Files added: $added_files"

if [ $added_files -gt 0 ]; then
  git add "$root_dir"
  git commit -m "Consolidate unique .py variants from all branches into $root_dir (added $added_files files)"
  echo "Committed consolidation."
else
  echo "No variant files to commit."
fi

# Show a short preview of the report
sed -n '1,160p' "$report"


