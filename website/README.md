# Project website

Quarto site for browsing results, training data, eval prompts, and model responses.

## Build locally

Requires [Quarto](https://quarto.org/docs/get-started/) and Python 3.11+.

```bash
pip install -r website/requirements.txt
quarto render website
```

Open `docs/index.html` in a browser, or run `quarto preview website`.

Site URL: `https://smpdl.github.io/safety-by-identity/`

Update `site-url` in `_quarto.yml` if the repo name or owner changes.
