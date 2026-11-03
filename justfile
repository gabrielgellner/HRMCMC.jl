# Development tasks for PolytopeSamplers.jl. Run `just` to list recipes.

# Julia apps (runic, jetls) are installed to ~/.julia/bin
export PATH := home_directory() / ".julia" / "bin" + ":" + env_var("PATH")

julia := "julia --startup-file=no"

# List recipes
default:
    @just --list

# Install the formatter and linter as Julia apps and instantiate all environments
setup:
    {{ julia }} -e 'using Pkg; Pkg.Apps.add("Runic"); Pkg.Apps.add(url = "https://github.com/aviatesk/JETLS.jl", rev = "release")'
    {{ julia }} --project -e 'using Pkg; Pkg.instantiate()'
    {{ julia }} --project=examples -e 'using Pkg; Pkg.instantiate()'

# Format all Julia files in place with Runic
fmt:
    runic --inplace .

# Check formatting without changing files, printing a diff
fmt-check:
    runic --check --diff .

# Static analysis with JETLS (JET inference plus lowering lints)
lint:
    jetls check -q .

# Run the test suite (Aqua, ExplicitImports, JET, AllocCheck and unit tests)
test:
    {{ julia }} --project -e 'using Pkg; Pkg.test()'

# Build the documentation into docs/build and open it
docs:
    {{ julia }} --project=docs docs/make.jl
    open docs/build/index.html

# Stress tests of convergence and hard geometry (about 10 minutes; not part of ci)
stress:
    {{ julia }} --project=test test/stress.jl

# Run every check (format, lint, tests); run before every push and merge
ci: fmt-check lint test

# Run the tests in a copy of the repository without its manifests, so that every environment
# is resolved from scratch as on CI and for users; this catches compat bounds that a local
# manifest hides. Run it before a release or after changing a dependency or a compat entry.
ci-fresh:
    #!/usr/bin/env bash
    set -euo pipefail
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    git ls-files -co --exclude-standard | rsync -a --files-from=- . "$tmp"
    cd "$tmp"
    {{ julia }} --project -e 'using Pkg; Pkg.instantiate()'
    {{ julia }} --project -e 'using Pkg; Pkg.test()'

# ── Release ───────────────────────────────────────────────────────────────────

# Step 1 — Preview the next version and the unreleased changelog (no files changed)
[group('release')]
release-1-preview:
    @echo "Next version: $(git-cliff --bumped-version)"
    @echo ""
    @git-cliff --bump --unreleased

# Step 2 — Start release/vX from origin/main and write CHANGELOG.md (pass VERSION to override)
[group('release')]
release-2-changelog VERSION='':
    #!/usr/bin/env bash
    # By default git-cliff determines the next version from the commit messages. Pass
    # VERSION to pin it, e.g. `just release-2-changelog 0.3.0`. Edit CHANGELOG.md by hand
    # after this step if needed: steps 3 and 4 read the section as edited.
    set -euo pipefail
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "error: the working tree has changes; commit or stash them first" >&2
        exit 1
    fi
    git fetch origin
    git switch --detach origin/main
    if [[ -n "{{ VERSION }}" ]]; then
        RAW="{{ VERSION }}"
        NEXT="v${RAW#v}"
        git switch -c "release/${NEXT}"
        echo "Writing changelog pinned to ${NEXT}..."
        git-cliff --unreleased --tag "${NEXT}" --prepend CHANGELOG.md
    else
        NEXT=$(git-cliff --bumped-version)
        git switch -c "release/${NEXT}"
        echo "Writing changelog for ${NEXT}..."
        git-cliff --unreleased --bump --prepend CHANGELOG.md
    fi
    echo ""
    echo "Review CHANGELOG.md, then run: just release-3-pr"

# Step 3 — Bump the version in Project.toml, commit, push release/vX and open its PR
[group('release')]
release-3-pr:
    #!/usr/bin/env bash
    set -euo pipefail
    BRANCH=$(git branch --show-current)
    if [[ "${BRANCH}" != release/v* ]]; then
        echo "error: not on a release/vX branch (run just release-2-changelog)" >&2
        exit 1
    fi
    VER="${BRANCH#release/}"
    sed -i.bak -E 's/^version = "[^"]+"/version = "'"${VER#v}"'"/' Project.toml && rm Project.toml.bak
    git add CHANGELOG.md Project.toml
    git commit -m "chore(release): ${VER}"
    git push -u origin "${BRANCH}"
    NOTES=$(just release-notes "${VER}")
    gh pr create --base main --head "${BRANCH}" --title "chore(release): ${VER}" --body "${NOTES}"
    echo "After just ci passes, run: just release-4-merge"

# Step 4 — Merge the release PR, tag its merge commit, create the GitHub release
[group('release')]
release-4-merge:
    #!/usr/bin/env bash
    set -euo pipefail
    BRANCH=$(git branch --show-current)
    if [[ "${BRANCH}" != release/v* ]]; then
        echo "error: not on a release/vX branch" >&2
        exit 1
    fi
    VER="${BRANCH#release/}"
    if [[ "$(gh pr view "${BRANCH}" --json state --jq .state)" != MERGED ]]; then
        gh pr merge "${BRANCH}" --merge --delete-branch=false
    fi
    # `gh pr merge` can exit 0 without merging; trust the state, not the exit code.
    STATE=$(gh pr view "${BRANCH}" --json state --jq .state)
    if [[ "${STATE}" != MERGED ]]; then
        echo "error: the ${BRANCH} PR is ${STATE}, not merged" >&2
        exit 1
    fi
    MERGE=$(gh pr view "${BRANCH}" --json mergeCommit --jq .mergeCommit.oid)
    git fetch origin
    git switch main
    git merge --ff-only origin/main
    git tag -a "${VER}" -m "${VER}" "${MERGE}"
    git push origin "${VER}"
    NOTES=$(just release-notes "${VER}")
    gh release create "${VER}" --verify-tag --title "${VER}" --notes "${NOTES}"
    git branch -D "${BRANCH}"
    if git ls-remote --exit-code --heads origin "${BRANCH}" >/dev/null; then
        git push origin --delete "${BRANCH}"
    fi
    echo "Released ${VER}"

# Steps 3 and 4 take their text from here, so edits made after step 2 are kept.
# Print a version's CHANGELOG.md section, as edited
[group('release')]
release-notes VERSION:
    #!/usr/bin/env bash
    set -euo pipefail
    VER="{{ VERSION }}"
    NOTES=$(awk -v head="## [${VER#v}]" 'index($0, head) == 1 {f = 1; next} /^## \[/ {f = 0} f' CHANGELOG.md | sed '/./,$!d')
    if [[ -z "${NOTES}" ]]; then
        echo "error: CHANGELOG.md has no section for ${VER#v}" >&2
        exit 1
    fi
    printf '%s\n' "${NOTES}"

# ── Examples ──────────────────────────────────────────────────────────────────

# Run examples/NAME.jl (e.g. just example constrained_foodweb)
[group('examples')]
example NAME:
    {{ julia }} --threads=auto --project=examples examples/{{ NAME }}.jl
