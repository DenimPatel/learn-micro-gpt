# Datasets

Two files, both committed, both small enough that downloading them at run time
would be a worse trade than not.

## `input.txt` — 32,033 names, 228 KB

The `names.txt` list from
[`karpathy/makemore`](https://github.com/karpathy/makemore) at commit `988aa59`.
**Byte-identical to upstream**, including the absence of a trailing newline:
sha256 `0a30b5557f192f32ab962680889aac5f6fda0f4cecf40a6d0b5694f58ea8cc4d`, checked
by `make verify-provenance`.

It is committed rather than fetched because `reference/microgpt.py` does this:

```python
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/...'
    urllib.request.urlretrieve(names_url, 'input.txt')
```

Two problems, both real:

- It resolves `input.txt` from the **current working directory**, not the script's
  directory. Run it from anywhere else and you get a fresh download — or, in a
  read-only directory, a crash.
- The URL points at a **mutable branch**. The bytes behind it can change at any
  time, which would silently invalidate every loss number recorded in this
  repository.

With the file committed, the `os.path.exists` check short-circuits, nothing is
downloaded, and the run is exactly the recorded one. It is also what lets the
browser playground work: `data/input.txt` is bundled into the app and written
into Pyodide's virtual filesystem before the reference runs.

`make run` copies it into a scratch directory and executes the reference there,
so the reference file itself is never modified.

## `shakespeare.txt` — 40,000 lines, 1.1 MB

The canonical "Tiny Shakespeare" character-level corpus, as used in
[`karpathy/char-rnn`](https://github.com/karpathy/char-rnn). Public domain.

**No v1 track uses it.** It is committed because the first thing anyone does
after finishing the names walkthrough is ask "what if I trained it on real
text?", and the answer should not be "go download something and possibly break
the reference". Point any of the five tracks at it with `--input`:

```sh
make run-python   # and then, to try it:
cd .run && python3 ../reference/microgpt.py
#   ... but that reads input.txt from the CWD, so copy the dataset in first
cp ../../data/shakespeare.txt input.txt
```

The tokenizer is derived from the dataset rather than configured, so it grows to
cover the new alphabet automatically and the model needs no changes. The
vocabulary goes from 27 to about 65, the embedding table and the output head
grow with it, and the character-level sequence length problem that `block_size =
16` papers over becomes obvious.

The `scaled` C config — `n_embd 256`, `n_layer 4`, `block_size 256` — exists
because of this file. At 40,000-line scale you need the context window, and the
contrast between the two configs is the most direct demonstration in the
repository of why a hyperparameter is a capacity decision.
