
# Installing Asparagus environment on Gefion


*Step 1*: Make a new environment
```
module load Python/3.11.3 SciPy-bundle/2024.05
mkdir ~/venv
cd ~/venv
python -m venv asparagus
cd ~/asparagus
source ~/venv/asparagus/bin/activate
```

*Step 2*: (Optional) Make it easy to load your environment

Add the following to the bottom of your `~/.bashrc`:
```
venv() {
    local base="/dcai/users/[USERNAME]/venv"
    [ -z "$1" ] && { printf "Usage: venv <name>\n"; return 1; }
    local dir="$base/$1"
    [ ! -d "$dir" ] && { printf "No such venv: %s\n" "$dir" >&2; return 1; }
    source "$dir/bin/activate"
}
```
where you insert your DCAI username instead of `[USERNAME]`.

Then to load your environment simply do `venv asparagus` (or any other environment name).

*Step 3*: You have to install torch before all other packakges:
```
cd ~/asparagus
```

*Step 4*: Install the rest of the dependencies in the following way
```
pip install -e . --no-deps
pip install . --group dcai
```

if pip does not have an argument called `--group`, you need to upgrade pip:
```
pip install --upgrade pip
```

## Searching for packages available on Gefion
If you want to install a package on Gefion, it can take a while to find a version which is allowed.

To make this easier, use

```
./scripts/find_gefion_pip_version.bash package-name
```
which will iterate through all versions available on Pypi.

_Note:_ Sometimes it fails because it cannot find a dependency, not the package itself, for a specific version of that package. This can often be solved by fixing the dependency also, so make sure to read the output carefully.