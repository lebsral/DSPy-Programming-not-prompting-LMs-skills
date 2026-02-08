# Example Projects

Full working DSPy projects that demonstrate the skills in this repo.

## Running Examples

Each example is a standalone project. To run one:

```bash
cd examples/<example-name>
pip install -U dspy
# Set your LM API key
export OPENAI_API_KEY="..."  # or whatever provider you use
python main.py
```

## Contributing Examples

When adding an example:

1. Create a directory under `examples/` with a descriptive name
2. Include a `main.py` entry point
3. Include a `README.md` explaining what it demonstrates
4. Keep dependencies minimal (just `dspy` + standard library where possible)
5. Use `dspy.LM("openai/gpt-4o-mini")` as the default LM (cheap and fast for demos)
