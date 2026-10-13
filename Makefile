.PHONY: test lint selftest demo bench clean

PY ?= python

test:          ## 全部单测
	$(PY) -m pytest tests/ -q

lint:          ## ruff 静态检查
	$(PY) -m ruff check rectforge tests examples

selftest:      ## G1 不变量自检
	$(PY) -m rectforge.cli selftest

demo:          ## 60 秒演示（全部门禁）
	$(PY) examples/run_demo.py

bench:         ## 完整基准 + 消融 + 确定性 -> benchmark.json
	$(PY) -m rectforge.cli run --out benchmark.json

densities:     ## 列出密度 + 归一化自检
	$(PY) -m rectforge.cli densities

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ examples/benchmark.json benchmark.json
