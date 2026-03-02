# SemBench

SemBench is a novel framework for evaluating the semantic capabilities of LLMs through their ability to generate definitions and examples. It only requires a **dictionary** as data. Comparison with the WiC dataset shows **strong correlations**, validating SemBench as a reliable evaluation approach.

## SemBench Evaluation

The **standard** SemBench evaluation (with example and definition generation) is:

```bash
python SemBenchScripts/LLMsWiCExampleVLLM.py \
    --modelname Llama3_8B \
    --k 5 \
    --language EN
```

The evaluation **without creating examples** (definition only) is:

```bash
python SemBenchScripts/LLMsWiCVLLM.py \
    --modelname Llama3_8B \
    --k 5 \
    --language EN
```

## WiC Evaluation

The scripts needed for WiC evaluation are in the `WiCScripts/` directory:

```bash
python WiCScripts/LLMsWiCVLLM.py \
    --modelname Llama3_8B \
    --k 5 \
    --language EN
```

