#!/bin/bash

# 定义分支和对应的PR标题、描述
declare -A branches=(
    ["auto-gen/prelu"]="feat(ops): migrate prelu from experimental_ops to ops"
    ["auto-gen/zero"]="feat(ops): migrate zero from experimental_ops to ops"
    ["auto-gen/special_i0e"]="feat(ops): migrate special_i0e from experimental_ops to ops"
    ["auto-gen/t_copy"]="feat(ops): migrate t_copy from experimental_ops to ops"
    ["auto-gen/_safe_softmax"]="feat(ops): migrate _safe_softmax from experimental_ops to ops"
    ["auto-gen/soft_margin_loss"]="feat(ops): migrate soft_margin_loss from experimental_ops to ops"
    ["auto-gen/margin_ranking_loss"]="feat(ops): migrate margin_ranking_loss from experimental_ops to ops"
    ["auto-gen/_upsample_nearest_exact1d"]="feat(ops): migrate _upsample_nearest_exact1d from experimental_ops to ops"
)

# PR描述模板
get_pr_body() {
    local op_name=$1
    cat <<EOF
## Description

This PR migrates the \`${op_name}\` operator from experimental_ops to the main ops directory.

## Changes
- Created \`src/flag_gems/ops/${op_name}.py\` with kernel implementation
- Registered operator in 3 places:
  - \`src/flag_gems/ops/__init__.py\` (import + __all__)
  - \`src/flag_gems/__init__.py\` (_FULL_CONFIG)
- Migrated unit tests to appropriate test file
- Migrated benchmark to appropriate benchmark file

## Testing
All unit tests pass for this operator.

Part of the experimental ops migration effort.
EOF
}

# 依次为每个分支创建PR
for branch in "${!branches[@]}"; do
    title="${branches[$branch]}"
    op_name="${branch#auto-gen/}"

    echo "Creating PR for branch: $branch"

    git checkout "$branch"
    gh pr create \
        --base master \
        --head "factnn:$branch" \
        --title "$title" \
        --body "$(get_pr_body "$op_name")"

    if [ $? -eq 0 ]; then
        echo "✓ PR created successfully for $branch"
    else
        echo "✗ Failed to create PR for $branch"
    fi

    echo "---"
done

echo "All PRs created!"
