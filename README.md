# semantic-navigator

[![Build Status](https://github.com/CouncilDataProject/semantic-navigator/workflows/CI/badge.svg)](https://github.com/CouncilDataProject/semantic-navigator/actions)
[![Documentation](https://github.com/CouncilDataProject/semantic-navigator/workflows/Documentation/badge.svg)](https://CouncilDataProject.github.io/semantic-navigator)

An active learning approach to query and search through large archival datasets.

---

## Installation

**Stable Release:** `pip install semantic-navigator`<br>
**Development Head:** `pip install git+https://github.com/CouncilDataProject/semantic-navigator.git`

## File Access in Python

The storage bucket is entirely public read (but protected write).
I uploaded an example file to a dev infrastructure.

```python
from gcsfs import GCSFileSystem

# Connect to the bucket with anonymous access
fs = GCSFileSystem("sem-nav-eva-005", token="anon")

# Read a remote file into memory
with fs.open("sem-nav-eva-005/Justfile", "r") as open_f:
    print(open_f.read())
```

This will print the contents of the file to the terminal.

If you can see where I am going, when we are trying to load text chunks for the app, instead of reading from local, we just need to make an `anon` connection to the project, then read the text file from whatever path is listed in the dataset row.

**NOTE: This isn't done just yet.**

i.e.

```python
from gcsfs import GCSFileSystem
import pandas as pd

# Connect to the bucket with anonymous access
fs = GCSFileSystem("sem-nav-eva-005", token="anon")

# Read the dataset
with fs.open("sem-nav-eva-005/dataset.parquet", "rb") as open_f:
    dataset = pd.read_parquet(open_f)

# Get random row from dataset
random_row_from_dataset = dataset.sample(1).iloc[0]

# Read the linked text example
with fs.open(random_row_from_dataset.chunk_text_path, "r") as open_f:
    example_random_text = open_f.read()

# Render w/ text
return render_template(
    "index.html",
    example_random_text=example_random_text,
)
```

## Documentation

For full package documentation please visit [CouncilDataProject.github.io/semantic-navigator](https://CouncilDataProject.github.io/semantic-navigator).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MPLv2 License**
