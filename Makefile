
.PHONY: test

test:
	OPTIMIZE=1 nosetests -v -s test/test_*.py
