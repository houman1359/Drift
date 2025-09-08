#!/usr/bin/env bash
set -euo pipefail

# Report per-branch merge status and blob-level differences for .py and .m files

cd "$(git rev-parse --show-toplevel)"

git fetch --all --prune --quiet || true

current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $current_branch"

# Build unified branch list (local + remote normalized)
git for-each-ref --format='%(refname:short)' refs/heads refs/remotes \
  | sed 's#^origin/##' \
  | sort -u > /tmp/_all_branches.txt

echo
echo "Blob-level comparison for *.py and *.m"

safe_list_file=/tmp/_safe_to_delete.txt
: > "$safe_list_file"

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

  echo
  echo "=== $br ==="

  mapfile -t files < <(git ls-tree -r --name-only "$ref" -- '*.py' '*.m' || true)
  if [ ${#files[@]} -eq 0 ]; then
    echo "(no .py or .m files)"
  fi

  diff_count=0
  new_count=0
  same_count=0

  for f in "${files[@]}"; do
    if git show "$current_branch:$f" >/dev/null 2>&1; then
      if git diff --quiet "$current_branch:$f" "$ref:$f"; then
        echo "SAME    $f"
        same_count=$((same_count+1))
      else
        echo "DIFF    $f"
        diff_count=$((diff_count+1))
      fi
    else
      echo "NEW     $f"
      new_count=$((new_count+1))
    fi
  done

  echo "Summary: SAME=$same_count DIFF=$diff_count NEW=$new_count"

  # Merge status
  if git merge-base --is-ancestor "$ref" "$current_branch"; then
    echo "Merge status: merged into $current_branch"
    if [ $diff_count -eq 0 ] && [ $new_count -eq 0 ]; then
      echo "SAFE_TO_DELETE: $br" | tee -a "$safe_list_file"
    fi
  else
    n=$(git rev-list --count "$current_branch..$ref" || echo 0)
    m=$(git rev-list --count "$ref..$current_branch" || echo 0)
    echo "Merge status: NOT merged (unique commits on $br: $n; on $current_branch: $m)"
  fi
done < /tmp/_all_branches.txt

echo
echo "Branches safe to delete (merged and no .py/.m differences):"
if [ -s "$safe_list_file" ]; then
  sed 's/^/ - /' "$safe_list_file"
else
  echo " - (none)"
fi


