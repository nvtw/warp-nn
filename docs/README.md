# Documentation

## Build documentation

### Install dependencies

```bash
pip install -e .[docs]
```

### Build the documentation

* Building docs once:

  ```bash
  cd docs
  sphinx-build . _build/html
  ```

* Building docs each time a file is changed:

  ```bash
  cd docs
  sphinx-autobuild . _build/html
  ```
